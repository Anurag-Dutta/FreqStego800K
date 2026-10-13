import os
import glob
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.cuda.amp import autocast
from sklearn.manifold import TSNE
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import sklearn
from packaging import version


plt.rcParams.update({
    'font.family': 'serif',
    'font.serif':  ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size':   9,
    'text.usetex': True,
})


def hex_to_rgb(h):
    return np.array([int(h[1:3],16), int(h[3:5],16), int(h[5:7],16)]) / 255.0


def rgb_to_hex(rgb):
    r, g, b = np.clip(rgb * 255, 0, 255).astype(int)
    return '#{:02X}{:02X}{:02X}'.format(r, g, b)


_anchors = [hex_to_rgb(c) for c in ('#FDE7A5', '#31C0AD', '#CDEBDF')]


def palette_color(t):
    t = np.clip(t, 0, 1)
    if t <= 0.5:
        return _anchors[0] + (_anchors[1] - _anchors[0]) * (t / 0.5)
    else:
        return _anchors[1] + (_anchors[2] - _anchors[1]) * ((t - 0.5) / 0.5)


def lighten(hx, f):
    r, g, b = int(hx[1:3],16), int(hx[3:5],16), int(hx[5:7],16)
    return '#{:02X}{:02X}{:02X}'.format(int(r+(255-r)*f),
                                         int(g+(255-g)*f),
                                         int(b+(255-b)*f))


def interpolate_color(hx1, hx2, t):
    r1, g1, b1 = hex_to_rgb(hx1) * 255
    r2, g2, b2 = hex_to_rgb(hx2) * 255
    return '#{:02X}{:02X}{:02X}'.format(int(r1+(r2-r1)*t),
                                         int(g1+(g2-g1)*t),
                                         int(b1+(b2-b1)*t))


_DS_HEX = [
    '#E63946', '#F4A261', '#2A9D8F', '#E9C46A',
    '#6A4C93', '#1982C4', '#8AC926', '#FF595E',
]


DS_INFO = [
    ('AI_ArtBench',          'AI-ArtBench',  _DS_HEX[0]),
    ('animal_faces_hq_afhq', 'AFHQ',         _DS_HEX[1]),
    ('BOSSbase_1.01',        'BOSSbase',      _DS_HEX[2]),
    ('BOWS2',                'BOWS2',         _DS_HEX[3]),
    ('emonet_faces',         'EmoNet-Face',   _DS_HEX[4]),
    ('happywhales',          'HappyWhales',   _DS_HEX[5]),
    ('landcover',            'LandCover',     _DS_HEX[6]),
    ('wikiart',              'WikiArt',       _DS_HEX[7]),
]
DS_COLOR_MAP = {k: (disp, col) for k, disp, col in DS_INFO}


ALGOS = [
    'dct_lsb', 'dct_phase', 'dct_zigzag',
    'dft_lsb', 'dft_phase', 'dft_radial',
    'dst_lsb', 'dwt_bp4', 'dwt_cross', 'dwt_lsb'
]
ALGO_TO_ID = {alg: i + 1 for i, alg in enumerate(ALGOS)}
ID_TO_ALGO  = {v: k for k, v in ALGO_TO_ID.items()}


ALGO_DISPLAY = {
    'cover'     : 'Cover',
    'dct_lsb'   : 'DCT-LSB',    'dct_phase' : 'DCT-Phase',
    'dct_zigzag': 'DCT-Zigzag', 'dft_lsb'   : 'DFT-LSB',
    'dft_phase' : 'DFT-Phase',  'dft_radial': 'DFT-Radial',
    'dst_lsb'   : 'DST-LSB',    'dwt_bp4'   : 'DWT-BP4',
    'dwt_cross' : 'DWT-Cross',  'dwt_lsb'   : 'DWT-LSB',
}


TASK1_COLORS = {0: '#FF9F1C', 1: '#2EC4B6'}
TASK1_NAMES  = {0: 'Cover',   1: 'Stego'}


CLASS_COLORS = [
    '#FF9F1C', '#FF3366', '#2EC4B6', '#FFE94E', '#3A86FF',
    '#FF6B6B', '#06D6A0', '#CB6CE6', '#FF9A3C', '#00B4D8', '#80ED99',
]
CLASS_NAMES = ['cover'] + ALGOS


BASE            = r"..."
GLOBAL_SAVE_DIR = os.path.join(BASE, 'checkpoints', 'vgg16_experiments')
TSNE_SAVE_DIR   = r"..."
FEAT_CACHE_DIR  = os.path.join(TSNE_SAVE_DIR, 'feature_cache')
os.makedirs(TSNE_SAVE_DIR,  exist_ok=True)
os.makedirs(FEAT_CACHE_DIR, exist_ok=True)


IMAGE_SIZE   = (224, 224)
BATCH_SIZE   = 32
NUM_WORKERS  = 0
SEED         = 42
MAX_SAMPLES  = 4000
POOL_MAX     = 100000


DATASETS = {
    'AI_ArtBench'          : (os.path.join(BASE, 'AI_ArtBench',          'Cover'), 'painting'),
    'animal_faces_hq_afhq' : (os.path.join(BASE, 'animal_faces_hq_afhq', 'Cover'), 'animal_face'),
    'BOSSbase_1.01'        : (os.path.join(BASE, 'BOSSbase_1.01',        'Cover'), 'natural_gray'),
    'BOWS2'                : (os.path.join(BASE, 'BOWS2',                'Cover'), 'natural_gray'),
    'emonet_faces'         : (os.path.join(BASE, 'EmoNet-Face',          'Cover'), 'face'),
    'happywhales'          : (os.path.join(BASE, 'happywhales',          'Cover'), 'wildlife'),
    'landcover'            : (os.path.join(BASE, 'landcover',            'Cover'), 'aerial'),
    'wikiart'              : (os.path.join(BASE, 'wikiart',              'Cover'), 'art'),
}


eval_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


_tsne_kwargs = dict(
    n_components=2, perplexity=40, learning_rate='auto',
    init='pca', random_state=SEED, n_jobs=-1
)
if version.parse(sklearn.__version__) >= version.parse('1.2'):
    _tsne_kwargs['max_iter'] = 1000
else:
    _tsne_kwargs['n_iter'] = 1000


print(f"sklearn {sklearn.__version__} -> "
      f"{'max_iter' if 'max_iter' in _tsne_kwargs else 'n_iter'}=1000")


class VGG16Scratch(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        def cbr(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.features = nn.Sequential(
            cbr(3,64),    cbr(64,64),   nn.MaxPool2d(2,2),
            cbr(64,128),  cbr(128,128), nn.MaxPool2d(2,2),
            cbr(128,256), cbr(256,256), cbr(256,256), nn.MaxPool2d(2,2),
            cbr(256,512), cbr(512,512), cbr(512,512), nn.MaxPool2d(2,2),
            cbr(512,512), cbr(512,512), cbr(512,512), nn.MaxPool2d(2,2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 4096), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(4096, 4096),    nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(4096, 1)
        )
    def forward(self, x):
        return self.classifier(self.features(x))



class VGG16ScratchMulti(nn.Module):
    def __init__(self, num_classes=11, dropout=0.5):
        super().__init__()
        def cbr(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.features = nn.Sequential(
            cbr(3,64),    cbr(64,64),   nn.MaxPool2d(2,2),
            cbr(64,128),  cbr(128,128), nn.MaxPool2d(2,2),
            cbr(128,256), cbr(256,256), cbr(256,256), nn.MaxPool2d(2,2),
            cbr(256,512), cbr(512,512), cbr(512,512), nn.MaxPool2d(2,2),
            cbr(512,512), cbr(512,512), cbr(512,512), nn.MaxPool2d(2,2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 4096), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(4096, 4096),    nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(4096, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))


def build_index_for_dataset(ds_name):
    cover_dir, domain = DATASETS[ds_name]
    EXT = ['*.jpg','*.jpeg','*.png','*.JPG','*.JPEG','*.PNG','*.pgm']
    cover_paths = []
    for ext in EXT:
        cover_paths.extend(glob.glob(os.path.join(cover_dir,'**',ext), recursive=True))
    cover_paths = sorted(list(dict.fromkeys(cover_paths)))

    cover_records = []
    stego_index   = defaultdict(dict)

    for p in cover_paths:
        rel    = os.path.relpath(p, cover_dir)
        img_id = rel.replace(os.sep, '_').rsplit('.', 1)[0]
        cover_records.append({'cover_path': p, 'image_id': img_id,
                               'dataset': ds_name, 'domain': domain})

    for alg in ALGOS:
        meta_path = os.path.join(BASE, ds_name, 'stego', alg, 'metadata.json')
        if not os.path.exists(meta_path):
            continue
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        for m in meta:
            if m.get('status') not in ('ok', 'warn_ber'):
                continue
            spa = m.get('spatial_path', '')
            if spa.startswith('./'):
                spa = os.path.join(BASE, ds_name, spa[2:])
            stego_index[m['image_id']][alg] = os.path.normpath(spa)

    valid = [r for r in cover_records if len(stego_index.get(r['image_id'], {})) > 0]
    return valid, stego_index


class FeatureExtractor(nn.Module):
    def __init__(self, features_block):
        super().__init__()
        self.features = features_block
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        return self.pool(self.features(x)).view(x.size(0), -1)



class FlatImageDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records   = records
        self.transform = transform
    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        r = self.records[idx]
        try:
            img = Image.open(r['path']).convert('RGB')
        except Exception:
            img = Image.new('RGB', IMAGE_SIZE, (0,0,0))
        if self.transform:
            img = self.transform(img)
        return img, r['label']


def cache_path(ds_name, task_id):
    return os.path.join(FEAT_CACHE_DIR, f'{ds_name}_task{task_id}.npz')


def save_cache(path, feats, labels):
    np.savez_compressed(path, feats=feats, labels=labels)
    print(f"  Cache saved : {path}")


def load_cache(path):
    data = np.load(path)
    print(f"  Cache loaded: {path}  ({data['feats'].shape[0]} samples)")
    return data['feats'], data['labels']


def extract_features(extractor, records, device, max_samples=None):
    if not records:
        return np.zeros((0, 512)), np.zeros(0)
    if max_samples and len(records) > max_samples:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(records), max_samples, replace=False)
        records = [records[i] for i in idx]
    records = [r for r in records if os.path.isfile(r['path'])]
    if not records:
        return np.zeros((0, 512)), np.zeros(0)

    loader = DataLoader(FlatImageDataset(records, eval_transform),
                        batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    extractor.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='  Extracting', ncols=80):
            imgs = imgs.to(device, non_blocking=True)
            with autocast():
                feats = extractor(imgs)
            all_feats.append(feats.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.vstack(all_feats), np.array(all_labels)


def build_task1_records(cover_records, stego_index, test_df):
    cover_map = {r['image_id']: r['cover_path'] for r in cover_records}
    csv_ids   = set(test_df['image_id'].astype(str).tolist())
    overlap   = csv_ids & set(cover_map.keys())

    records = []
    if len(overlap) > 0:
        for img_id in overlap:
            records.append({'path': cover_map[img_id], 'label': 0})
            pairs = list(stego_index.get(img_id, {}).items())
            if pairs:
                records.append({'path': pairs[0][1], 'label': 1})
    else:
        stem_to_stego = {}
        for img_id, ad in stego_index.items():
            stem = img_id.rsplit('_', 1)[-1] if '_' in img_id else img_id
            stem_to_stego[stem] = list(ad.items())
        for _, row in test_df.iterrows():
            cpath = str(row['cover_path'])
            if not os.path.isfile(cpath):
                continue
            records.append({'path': cpath, 'label': 0})
            stem  = os.path.splitext(os.path.basename(cpath))[0]
            pairs = list(stego_index.get(str(row['image_id']), {}).items())
            if not pairs:
                pairs = stem_to_stego.get(stem, [])
            if pairs:
                records.append({'path': pairs[0][1], 'label': 1})
    return records


def draw_tsne_panel(ax, emb, labels, colors, names, task_id):
    for lbl in sorted(np.unique(labels)):
        mask      = (labels == lbl)
        color     = colors[lbl] if isinstance(colors, dict) else colors[int(lbl)]
        r, g, b   = hex_to_rgb(color)
        face_rgba = (r, g, b, 0.55)
        edge_rgba = (0.0, 0.0, 0.0, 0.90)
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            facecolors = face_rgba,
            edgecolors = edge_rgba,
            linewidths = 0.05,
            s          = 2,
            rasterized = True,
            zorder     = 3,
        )

    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')
        spine.set_linewidth(0.7)

    ax.tick_params(axis='both', colors='#555555', labelsize=6)
    ax.grid(True, color='#EEEEEE', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)


def pool_and_cap(feat_list, label_list, cap):
    if not feat_list:
        return None, None
    F = np.vstack(feat_list)
    L = np.concatenate(label_list)
    if len(F) > cap:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(F), cap, replace=False)
        F, L = F[idx], L[idx]
    print(f"  Pooled: {F.shape[0]} samples x {F.shape[1]} dims")
    return F, L


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  sklearn: {sklearn.__version__}")

    all_feats_t1, all_labels_t1 = [], []
    all_feats_t2, all_labels_t2 = [], []

    for ds_name in [d[0] for d in DS_INFO]:
        save_dir = os.path.join(GLOBAL_SAVE_DIR, ds_name)
        print(f"\n{'='*55}\n  {ds_name}\n{'='*55}")

        cp_t1    = cache_path(ds_name, 1)
        t1_model = os.path.join(save_dir, 'task1_best_model.pth')
        t1_split = os.path.join(save_dir, 'task1_test_cover_split.csv')

        if os.path.exists(cp_t1):
            f, l = load_cache(cp_t1)
        elif os.path.exists(t1_model) and os.path.exists(t1_split):
            model_t1 = VGG16Scratch(dropout=0.5)
            model_t1.load_state_dict(torch.load(t1_model, map_location=device))
            model_t1.to(device)
            ext_t1 = FeatureExtractor(model_t1.features).to(device)
            cover_records, stego_index = build_index_for_dataset(ds_name)
            records_t1 = build_task1_records(
                cover_records, stego_index, pd.read_csv(t1_split))
            f, l = extract_features(ext_t1, records_t1, device, MAX_SAMPLES)
            if len(f):
                save_cache(cp_t1, f, l)
            del model_t1, ext_t1
            torch.cuda.empty_cache()
        else:
            print(f"  [SKIP T1]")
            f, l = np.zeros((0,512)), np.zeros(0)

        if len(f):
            all_feats_t1.append(f)
            all_labels_t1.append(l)

        cp_t2    = cache_path(ds_name, 2)
        t2_model = os.path.join(save_dir, 'task2_best_model.pth')
        t2_split = os.path.join(save_dir, 'task2_test_split.csv')

        if os.path.exists(cp_t2):
            f, l = load_cache(cp_t2)
        elif os.path.exists(t2_model) and os.path.exists(t2_split):
            model_t2 = VGG16ScratchMulti(num_classes=11, dropout=0.5)
            model_t2.load_state_dict(torch.load(t2_model, map_location=device))
            model_t2.to(device)
            ext_t2 = FeatureExtractor(model_t2.features).to(device)
            records_t2 = pd.read_csv(t2_split)[['path','label']].to_dict('records')
            f, l = extract_features(ext_t2, records_t2, device, MAX_SAMPLES)
            if len(f):
                save_cache(cp_t2, f, l)
            del model_t2, ext_t2
            torch.cuda.empty_cache()
        else:
            print(f"  [SKIP T2]")
            f, l = np.zeros((0,512)), np.zeros(0)

        if len(f):
            all_feats_t2.append(f)
            all_labels_t2.append(l)

    print(f"\n{'='*55}\n  Pooling\n{'='*55}")
    feats_t1, labels_t1 = pool_and_cap(all_feats_t1, all_labels_t1, POOL_MAX)
    feats_t2, labels_t2 = pool_and_cap(all_feats_t2, all_labels_t2, POOL_MAX)

    def run_tsne(feats, tag):
        print(f"  t-SNE [{tag}] on {feats.shape} ...")
        return TSNE(**_tsne_kwargs).fit_transform(feats)

    emb_t1 = run_tsne(feats_t1, 'Task1') if feats_t1 is not None else None
    emb_t2 = run_tsne(feats_t2, 'Task2') if feats_t2 is not None else None

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(4, 1.5),
        facecolor='white',
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.06, right=0.99,
        bottom=0.08, top=0.88,
        wspace=0.22,
    )
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')

    if emb_t1 is not None:
        draw_tsne_panel(ax1, emb_t1, labels_t1, TASK1_COLORS, TASK1_NAMES, 1)
    else:
        ax1.text(0.5, 0.5, 'No data', ha='center', va='center',
                 transform=ax1.transAxes)

    if emb_t2 is not None:
        draw_tsne_panel(ax2, emb_t2, labels_t2, CLASS_COLORS, CLASS_NAMES, 2)
    else:
        ax2.text(0.5, 0.5, 'No data', ha='center', va='center',
                 transform=ax2.transAxes)

    ax1.set_title(r'\textit{Steganography Detection}',
                  fontsize=10, pad=5, color='#111111')
    ax2.set_title(r'\textit{Algorithm Attribution}',
                  fontsize=10, pad=5, color='#111111')

    base = os.path.join(TSNE_SAVE_DIR, 'tsne_combined')
    with PdfPages(base + '.pdf') as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(base + '.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"\nSaved:\n  {base}.pdf\n  {base}.png")
    print(f"Feature cache in: {FEAT_CACHE_DIR}")