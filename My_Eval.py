# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import utils
import argparse
from qm9 import dataset
from qm9.models import get_model, get_autoencoder, get_latent_diffusion
import os
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked
import torch
import time
import pickle
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9.sampling import sample
from qm9.analyze import analyze_stability_for_molecules, analyze_node_distribution
from qm9.utils import prepare_context, compute_mean_mad
from qm9 import visualizer as qm9_visualizer
import qm9.losses as losses
from reconstruct import reconstruct_from_generated

try:
    from qm9 import rdkit_functions
except ModuleNotFoundError:
    print('Not importing rdkit functions.')

from rdkit import Chem
from rdkit.Chem import AllChem
import torch


# adjust if your one-hot order is different
ATOM_TYPES =['H', 'B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br', 'I', 'Hg', 'Bi'] 

def one_hot_to_atomic_number(one_hot_row):
    """Convert one-hot vector to atomic number."""
    idx = one_hot_row.argmax()
    symbol = ATOM_TYPES[idx]
    return Chem.PeriodicTable.GetAtomicNumber(Chem.GetPeriodicTable(), symbol)


def make_rdkit_mol(atom_numbers, coordinates):
    """
    Create an RDKit Mol object from atomic numbers and coordinates.
    """
    mol = Chem.RWMol()
    atom_ids = []
    for Z in atom_numbers:
        atom = Chem.Atom(int(Z))
        atom_ids.append(mol.AddAtom(atom))
    
    mol = mol.GetMol()
    conf = Chem.Conformer(len(atom_numbers))
    for i, coord in enumerate(coordinates):
        conf.SetAtomPosition(i, coord.tolist())
    
    mol.AddConformer(conf)
    return mol


def calculate_mmff_energies(molecules, verbose=False):
    """
    Calculates MMFF conformer energy for a batch of GeoLDM-generated molecules.

    Parameters
    ----------
    molecules : dict with keys
        'one_hot': Tensor [B, N, T]
        'x':       Tensor [B, N, 3]   (coordinates)
        'node_mask': Tensor [B, N, 1] or [B, N]
    
    Returns
    -------
    energies : list[float]
        MMFF energy for each valid molecule. If failed, returns None for that molecule.
    """
    one_hot = molecules["one_hot"]
    x = molecules["x"]
    node_mask = molecules["node_mask"]

    if isinstance(one_hot, list): one_hot = torch.stack(one_hot)
    if isinstance(x, list): x = torch.stack(x)
    if isinstance(node_mask, list): node_mask = torch.stack(node_mask)

    if node_mask.dim() == 3:
        node_mask = node_mask.squeeze(-1)

    one_hot = one_hot.detach().cpu().numpy()
    x = x.detach().cpu().numpy()
    node_mask = node_mask.detach().cpu().numpy()

    energies = []
    for oh, xyz, mask in zip(one_hot, x, node_mask):
        try:
            oh_valid = oh[mask.astype(bool)]
            xyz_valid = xyz[mask.astype(bool)]
            atom_numbers = [one_hot_to_atomic_number(v) for v in oh_valid]

            # mol = make_rdkit_mol(atom_numbers, xyz_valid)
            mol = reconstruct_from_generated(xyz, atom_numbers)
            # mol = Chem.AddHs(mol)
            Chem.SanitizeMol(mol)

            props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94')
            ff = AllChem.MMFFGetMoleculeForceField(mol, props)
            energy = ff.CalcEnergy()
            energies.append(energy)

        except Exception as e:
            if verbose:
                print(f"Failed molecule: {e}")
            energies.append(None)

    return energies

def check_mask_correct(variables, node_mask):
    for variable in variables:
        assert_correctly_masked(variable, node_mask)
import numpy as np


def average_size_without_H(molecules, h_index: int = 1) -> float:
    """
    Compute the mean heavy‑atom (non‑H) count per molecule.

    Works for:
        one_hot   : list[tensor]  *or*  tensor shaped (B, N, T) or (B, T, N)
        node_mask : tensor shaped (B, N) or (B, N, 1)
    
    """
    ATOM_TYPES =['H', 'B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br', 'I', 'Hg', 'Bi']       
    # ────────────────────────────── node_mask ────────────────────────────────
    node_mask = molecules["node_mask"]                       # (B, N) or (B, N, 1)

    # If the mask has an extra trailing 1‑dim, remove it -> (B, N)
    if node_mask.dim() == 3 and node_mask.shape[-1] == 1:
        node_mask = node_mask.squeeze(-1)

    node_mask = node_mask.bool()                             # ensure bool

    # ─────────────────────────────── one_hot ────────────────────────────────
    one_hot = molecules["one_hot"]
    if isinstance(one_hot, list):                            # list → tensor
        one_hot = torch.stack(one_hot, dim=0)                # (B, ?, ?)

    # Make layout (B, N, T) so node dimension lines up with node_mask
    if one_hot.shape[1] != node_mask.shape[1] and one_hot.shape[2] == node_mask.shape[1]:
        one_hot = one_hot.permute(0, 2, 1)                   # swap to (B, N, T)

    # ─────────────────────── heavy‑atom logical mask ─────────────────────────
    is_h = one_hot[..., h_index] == 1                        # (B, N)
    heavy_nodes = (~is_h) & node_mask                        # (B, N)

    # ──────────────────────────── average size ──────────────────────────────
    heavy_counts = heavy_nodes.sum(dim=1, dtype=torch.float32)  # (B,)
    return heavy_counts.mean().item()

def analyze_and_save(args, eval_args, device, generative_model,
                     nodes_dist, prop_dist, dataset_info, n_samples=10,
                     batch_size=10, save_to_xyz=False, save_samples_to=None, load_samples=None):
    batch_size = min(batch_size, n_samples)
    if load_samples:
        molecules = torch.load(load_samples, map_location='cpu')
        print(f"✅  Loaded {molecules['one_hot'].shape[0]} samples "
              f"from {load_samples}")
       # print(molecules['node_mask'])
    else:
        
        assert n_samples % batch_size == 0
        molecules = {'one_hot': [], 'x': [], 'node_mask': []}
        start_time = time.time()
        with torch.no_grad():
            for i in range(int(n_samples/batch_size)):
                nodesxsample = nodes_dist.sample(batch_size)
                one_hot, charges, x, node_mask = sample(
                    args, device, generative_model, dataset_info, prop_dist=prop_dist, nodesxsample=nodesxsample)
        
                molecules['one_hot'].append(one_hot.cpu())
                molecules['x'].append(x.cpu())
                molecules['node_mask'].append(node_mask.cpu())
        
                current_num_samples = (i+1) * batch_size
                secs_per_sample = (time.time() - start_time) / current_num_samples
                print('\t %d/%d Molecules generated at %.2f secs/sample' % (
                    current_num_samples, n_samples, secs_per_sample))
        
                if save_to_xyz:
                    id_from = i * batch_size
                    qm9_visualizer.save_xyz_file(
                        join(eval_args.model_path, 'eval/analyzed_molecules/'),
                        one_hot, charges, x, dataset_info, id_from, name='molecule',
                        node_mask=node_mask)

        molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}

    energies = calculate_mmff_energies(molecules)
    # Calculate and print the average energy
    valid_energies = [e for e in energies if e is not None]
    if valid_energies:
        average_energy = sum(valid_energies) / len(valid_energies)
        print(f'Average MMFF Conformer Energy: {average_energy:.2f} kcal/mol')
    else:
        print('No valid energies to compute average.')

    stability_dict, rdkit_metrics = analyze_stability_for_molecules(
        molecules, dataset_info)
    
    # Typical GeoLDM / QM9 order:  [H, C, N, O, F]
    ATOM_TYPES =['H', 'B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br', 'I', 'Hg', 'Bi']         # one‑hot column → element
    H_INDEX = ATOM_TYPES.index('H')                 # column that encodes hydrogen
    Avg_size=average_size_without_H(molecules,H_INDEX)
    if save_samples_to:
            torch.save(molecules, save_samples_to)
            print(f"💾  Samples written to {save_samples_to}")

    return stability_dict, rdkit_metrics, Avg_size


def test(args, flow_dp, nodes_dist, device, dtype, loader, partition='Test', num_passes=1):
    flow_dp.eval()
    nll_epoch = 0
    n_samples = 0
    for pass_number in range(num_passes):
        with torch.no_grad():
            for i, data in enumerate(loader):
                # Get data
                x = data['positions'].to(device, dtype)
                node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
                edge_mask = data['edge_mask'].to(device, dtype)
                one_hot = data['one_hot'].to(device, dtype)
                charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

                batch_size = x.size(0)

                x = remove_mean_with_mask(x, node_mask)
                check_mask_correct([x, one_hot], node_mask)
                assert_mean_zero_with_mask(x, node_mask)

                h = {'categorical': one_hot, 'integer': charges}

                if len(args.conditioning) > 0:
                    context = prepare_context(args.conditioning, data).to(device, dtype)
                    assert_correctly_masked(context, node_mask)
                else:
                    context = None

                # transform batch through flow
                nll, _, _ = losses.compute_loss_and_nll(args, flow_dp, nodes_dist, x, h, node_mask,
                                                        edge_mask, context)
                # standard nll from forward KL

                nll_epoch += nll.item() * batch_size
                n_samples += batch_size
                if i % args.n_report_steps == 0:
                    print(f"\r {partition} NLL \t, iter: {i}/{len(loader)}, "
                          f"NLL: {nll_epoch/n_samples:.2f}")

    return nll_epoch/n_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="outputs/edm_1",
                        help='Specify model path')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Specify model path')
    parser.add_argument('--batch_size_gen', type=int, default=100,
                        help='Specify model path')
    parser.add_argument('--save_to_xyz', type=eval, default=False,
                        help='Should save samples to xyz files.')
    # NEW ≫ optional persistence flags
    parser.add_argument('--save_samples_to', default=None,
                   help='Pickle file to write generated tensors')
    parser.add_argument('--load_samples',    default=None,
                   help='Pickle file to read instead of generating')

    eval_args, unparsed_args = parser.parse_known_args()

    assert eval_args.model_path is not None
    
        

    with open(join(eval_args.model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    # CAREFUL with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = 1
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = 'sum'

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    dtype = torch.float32
    utils.create_folders(args)
    print(args)

    # Retrieve QM9 dataloaders
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

    dataset_info = get_dataset_info(args.dataset, args.remove_h)

    # Load model
    generative_model, nodes_dist, prop_dist = get_latent_diffusion(args, device, dataset_info, dataloaders['train'])
    if prop_dist is not None:
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
        prop_dist.set_normalizer(property_norms)
    generative_model.to(device)

    fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
    flow_state_dict = torch.load(join(eval_args.model_path, fn), map_location=device)
    generative_model.load_state_dict(flow_state_dict)

    # Analyze stability, validity, uniqueness and novelty
 
    stability_dict, rdkit_metrics, Avg_size = analyze_and_save(
        args, eval_args, device, generative_model, nodes_dist,
        prop_dist, dataset_info, n_samples=eval_args.n_samples,
        batch_size=eval_args.batch_size_gen, save_to_xyz=eval_args.save_to_xyz, save_samples_to=eval_args.save_samples_to, load_samples=eval_args.load_samples)
    print(stability_dict)

    if rdkit_metrics is not None:
        rdkit_metrics = rdkit_metrics[0]
        print("Validity %.4f, Uniqueness: %.4f, Novelty: %.4f" % (rdkit_metrics[0], rdkit_metrics[1], rdkit_metrics[2]))
    else:
        print("Install rdkit roolkit to obtain Validity, Uniqueness, Novelty")

    # In GEOM-Drugs the validation partition is named 'val', not 'valid'.
    if args.dataset == 'geom':
        val_name = 'val'
        num_passes = 1
    else:
        val_name = 'valid'
        num_passes = 5
    #NEW
    print( stability_dict)
    print(Avg_size)
    # Evaluate negative log-likelihood for the validation and test partitions
    '''val_nll = test(args, generative_model, nodes_dist, device, dtype,
                   dataloaders[val_name],
                   partition='Val')
    print(f'Final val nll {val_nll}')
    test_nll = test(args, generative_model, nodes_dist, device, dtype,
                    dataloaders['test'],
                    partition='Test', num_passes=num_passes)
    print(f'Final test nll {test_nll}')

    print(f'Overview: val nll {val_nll} test nll {test_nll}', stability_dict)
    with open(join(eval_args.model_path, 'eval_log.txt'), 'w') as f:
        print(f'Overview: val nll {val_nll} test nll {test_nll}',
              stability_dict,
              file=f)'''


if __name__ == "__main__":
    main()