import numpy as np
import awkward as ak
import vector
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm

# Define functions needed to transform the input variables
def _p4_from_pxpypze(px, py, pz, energy):
    import vector
    vector.register_awkward()
    return vector.zip({'px': px, 'py': py, 'pz': pz, 'energy': energy})

def _transform(dataframe, start=0, stop=-1):

    df = dataframe.iloc[start:stop]

    def _col_list(prefix, max_particles=200):
        return ['%s_%d' % (prefix, i) for i in range(max_particles)]

    _px = df[_col_list('PX')].values
    _py = df[_col_list('PY')].values
    _pz = df[_col_list('PZ')].values
    _e = df[_col_list('E')].values

    mask = _e > 0
    n_particles = np.sum(mask, axis=1)

    px = ak.unflatten(_px[mask], n_particles)
    py = ak.unflatten(_py[mask], n_particles)
    pz = ak.unflatten(_pz[mask], n_particles)
    energy = ak.unflatten(_e[mask], n_particles)

    p4 = _p4_from_pxpypze(px, py, pz, energy)

    jet_p4 = ak.sum(p4, axis=1)

    # outputs
    v = {}
    v['label'] = df['is_signal_new'].values

    v['jet_pt'] = jet_p4.pt
    v['jet_eta'] = jet_p4.eta
    v['jet_phi'] = jet_p4.phi
    v['jet_energy'] = jet_p4.energy
    v['jet_mass'] = jet_p4.mass
    v['jet_nparticles'] = n_particles

    v['part_px'] = px
    v['part_py'] = py
    v['part_pz'] = pz
    v['part_energy'] = energy

    _jet_etasign = ak.to_numpy(np.sign(v['jet_eta']))
    _jet_etasign[_jet_etasign == 0] = 1
    v['part_deta'] = (p4.eta - v['jet_eta']) * _jet_etasign
    v['part_dphi'] = p4.deltaphi(jet_p4)

    return v

# Padding function
def _pad(a, maxlen, value=0, dtype='float32'):
    if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
        return a
    elif isinstance(a, ak.Array):
        if a.ndim == 1:
            a = ak.unflatten(a, 1)
        a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
        return ak.values_astype(a, dtype)
    else:
        x = (np.ones((len(a), maxlen)) * value).astype(dtype)
        for idx, s in enumerate(a):
            if not len(s):
                continue
            trunc = s[:maxlen].astype(dtype)
            x[idx, :len(trunc)] = trunc
        return x

# Create input for the model
def build_features_and_labels(a, transform_features=True):
    # Compute new features
    a['part_mask'] = ak.ones_like(a['part_deta'])
    a['part_pt'] = np.hypot(a['part_px'], a['part_py'])
    a['part_pt_log'] = np.log(a['part_pt'])
    a['part_e_log'] = np.log(a['part_energy'])
    a['part_logptrel'] = np.log(a['part_pt']/a['jet_pt'])
    a['part_logerel'] = np.log(a['part_energy']/a['jet_energy'])
    a['part_deltaR'] = np.hypot(a['part_deta'], a['part_dphi'])
    a['jet_isTop'] = a['label']
    a['jet_isQCD'] = 1-a['label']

    if transform_features:
        a['part_pt_log'] = (a['part_pt_log'] - 1.7) * 0.7
        a['part_e_log'] = (a['part_e_log'] - 2.0) * 0.7
        a['part_logptrel'] = (a['part_logptrel'] - (-4.7)) * 0.7
        a['part_logerel'] = (a['part_logerel'] - (-4.7)) * 0.7
        a['part_deltaR'] = (a['part_deltaR'] - 0.2) * 4.0

    feature_list = {
        'pf_points': ['part_deta', 'part_dphi'], # not used in ParT
        'pf_features': [
            'part_pt_log',
            'part_e_log',
            'part_logptrel',
            'part_logerel',
            'part_deltaR',
            'part_deta',
            'part_dphi',
        ],
        'pf_vectors': [
            'part_px',
            'part_py',
            'part_pz',
            'part_energy',
        ],
        'pf_mask': ['part_mask']
    }

    out = {}
    for k, names in feature_list.items():
        out[k] = np.stack([_pad(a[n], maxlen=128).to_numpy() for n in names], axis=1)
    label_list = ['jet_isTop', 'jet_isQCD']
    out['label'] = np.stack([a[n].to_numpy().astype('int') for n in label_list], axis=1)

    return out

# Define data configuration
class DataConfig:
    def __init__(self):
        self.input_dicts = {
            'pf_features': [
            'part_pt_log',
            'part_e_log',
            'part_logptrel',
            'part_logerel',
            'part_deltaR',
            'part_deta',
            'part_dphi',
            ]
        }
        self.label_value = ['jet_isTop', 'jet_isQCD']
        self.input_names = ['pf_points', 'pf_features', 'pf_vectors', 'pf_mask']
        self.input_shapes = {
            'pf_points': (128, 2),
            'pf_features': (128, 7),
            'pf_vectors': (128, 4),
            'pf_mask': (128, 1)
        }

# Function to train model
def train_model(model, filename, train_loader, val_loader, optimizer, num_epochs, device='cuda'):

    model.to(device)
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss, correct, total = 0, 0, 0

        for points, features, lorentz_vectors, mask, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} Training'):
            points, features, lorentz_vectors, mask, labels = (
                points.to(device),
                features.to(device),
                lorentz_vectors.to(device),
                mask.to(device),
                labels.to(device),
            )

            optimizer.zero_grad()
            outputs = model(points, features, lorentz_vectors, mask)

            # Transform one-hote encoded labels to class indices
            labels = torch.argmax(labels, dim=1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for points, features, lorentz_vectors, mask, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} Validation'):
                points, features, lorentz_vectors, mask, labels = (
                    points.to(device),
                    features.to(device),
                    lorentz_vectors.to(device),
                    mask.to(device),
                    labels.to(device),
                )

                outputs = model(points, features, lorentz_vectors, mask)

                # Transform one-hote encoded labels to class indices
                labels = torch.argmax(labels, dim=1)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        val_loss /= len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    print("Training complete.")
    torch.save(model.state_dict(), filename)
    model.load_state_dict(torch.load(filename))
    return model, history

# Plot training history
def plot_training_history(history):

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Loss plot
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Training & Validation Loss")
    ax1.xaxis.set_major_locator(MultipleLocator(2))

    # Accuracy plot
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["val_acc"], label="Val Acc")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.set_title("Training & Validation Accuracy")
    ax2.xaxis.set_major_locator(MultipleLocator(2))

    plt.tight_layout()
    plt.savefig('history.pdf')
    plt.show()

# Evaluate model on test set
def evaluate_model(model, dataloader, device='cuda'):

    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)

    model.eval()
    model.to(device)
    test_loss, correct, total = 0, 0, 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for points, features, lorentz_vectors, mask, labels in dataloader:
            points, features, lorentz_vectors, mask, labels = (
                points.to(device),
                features.to(device),
                lorentz_vectors.to(device),
                mask.to(device),
                labels.to(device),
            )

            outputs = model(points, features, lorentz_vectors, mask)

            # Transform one-hot encoded labels to class indices
            labels = torch.argmax(labels, dim=1)

            # Transform logits to probabilities
            probs = softmax(outputs)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    test_acc = correct / total
    print(f"Test Acc: {test_acc:.4f}")
    test_loss /= len(dataloader)
    print(f"Test Loss: {test_loss:.4f}")

    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)

# Plotting ROC curves and determining rjection rate
def plot_roc_curves_vs_background(y_true, y_probs, class_labels, background_class_label, tpr_target=0.5, figsize=(8, 6)):
    background_idx = class_labels.index(background_class_label)
    plt.figure(figsize=figsize)

    for i, class_label in enumerate(class_labels):
        if class_label == background_class_label:
            continue  # Skip background

        # Mask for signal and background only
        mask = (np.array(y_true) == i) | (np.array(y_true) == background_idx)
        y_true_bin = np.array([1 if y == i else 0 for y in y_true])[mask]
        print(y_true_bin)
        y_score = y_probs[:, i][mask]

        # ROC
        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        auc = roc_auc_score(y_true_bin, y_score)

        # Rejection point
        idx = np.argmin(np.abs(tpr - tpr_target))
        fpr_at_target = fpr[idx]
        tpr_at_target = tpr[idx]
        rej = 1 / fpr_at_target if fpr_at_target > 0 else float("inf")

        label = f"{class_label} (AUC={auc:.4f}, Rej@{int(tpr_target*100)}%={rej:.0f})"
        plt.plot(fpr, tpr, label=label)
        plt.plot(fpr_at_target, tpr_at_target, "o", markersize=5)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curves: Signal vs. Background")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
