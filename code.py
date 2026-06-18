"""
=============================================================
  Détection & Lissage + Suppression Arrière-plan — v3
  Interface Streamlit
=============================================================
Dépendances obligatoires :
    pip install streamlit numpy opencv-python scipy matplotlib pillow

Dépendance optionnelle (meilleure qualité IA) :
    pip install rembg onnxruntime

Lancement :
    streamlit run code.py
=============================================================
"""

import io
import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter, convolve, median_filter
from scipy.interpolate import RectBivariateSpline
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import scipy.sparse as sc
from scipy.sparse.linalg import spsolve
import zipfile

try:
    from rembg import remove as rembg_remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False


# ══════════════════════════════════════════════════════════════
#  UTILITAIRES
# ══════════════════════════════════════════════════════════════

def normalize(img):
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn + 1e-8)

def to_uint8(img):
    return (np.clip(normalize(img), 0, 1) * 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════
#  PRÉ-TRAITEMENT & LISSAGE
# ══════════════════════════════════════════════════════════════

def clahe_enhance(img):
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(to_uint8(img)).astype(np.float64) / 255.0

def gaussian_smoothing(img, sigma=1.5):
    return gaussian_filter(img, sigma=sigma)

def bilateral_smoothing(img, sigma_color=0.1, sigma_space=10):
    r = cv2.bilateralFilter(to_uint8(img), 9, sigma_color*255, sigma_space)
    return r.astype(np.float64) / 255.0

def median_smoothing(img, size=5):
    return median_filter(img, size=max(3, int(size)|1)).astype(np.float64)

def spline_smooth(img, scale=0.5):
    rows, cols = img.shape
    y0 = np.arange(rows, dtype=np.float64)
    x0 = np.arange(cols, dtype=np.float64)
    ys = np.linspace(0, rows-1, max(6, int(rows*scale)))
    xs = np.linspace(0, cols-1, max(6, int(cols*scale)))
    sub = img[np.ix_(np.round(ys).astype(int), np.round(xs).astype(int))]
    spl = RectBivariateSpline(ys, xs, sub, kx=3, ky=3)
    return np.clip(spl(y0, x0), 0, 1)


# ══════════════════════════════════════════════════════════════
#  DÉTECTION DE CONTOURS
# ══════════════════════════════════════════════════════════════

def _nms(mag, direction):
    rows, cols = mag.shape
    out = np.zeros_like(mag)
    angle = np.rad2deg(direction % np.pi)
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            a = angle[i, j]; m = mag[i, j]
            if   (0<=a<22.5) or (157.5<=a<=180): q,r = mag[i,j+1], mag[i,j-1]
            elif  22.5<=a<67.5:                  q,r = mag[i+1,j-1], mag[i-1,j+1]
            elif  67.5<=a<112.5:                 q,r = mag[i+1,j], mag[i-1,j]
            else:                                q,r = mag[i-1,j-1], mag[i+1,j+1]
            out[i,j] = m if (m>=q and m>=r) else 0.0
    return out

def sobel_nms(img):
    Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float64)
    Gx = convolve(img, Kx); Gy = convolve(img, Kx.T)
    mag = normalize(np.hypot(Gx, Gy))
    return normalize(_nms(mag, np.arctan2(Gy, Gx)))

def canny_edges(img, low=0.05, high=0.18):
    lo, hi = int(low*255), int(high*255)
    return cv2.Canny(to_uint8(img), lo, hi, apertureSize=3, L2gradient=True).astype(np.float64)/255.0

def log_zero_crossing(img, sigma=1.5):
    lap = convolve(gaussian_filter(img, sigma), np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float64))
    rows, cols = lap.shape; zc = np.zeros((rows, cols))
    for dy, dx in [(0,1),(1,0)]:
        a = lap[:-dy or rows, :-dx or cols]; b = lap[dy:, dx:]
        zc[:-dy or rows, :-dx or cols] += ((a*b)<0).astype(np.float64)
    return np.clip(zc, 0, 1)

def spline_edges(img, scale=0.5):
    gy, gx = np.gradient(spline_smooth(img, scale))
    return normalize(np.hypot(gx, gy))

def _butterworth(shape, cutoff, order, high_pass):
    rows, cols = shape; cy, cx = rows//2, cols//2
    Y, X = np.ogrid[:rows, :cols]
    D = np.sqrt(((Y-cy)/(rows*cutoff+1e-6))**2 + ((X-cx)/(cols*cutoff+1e-6))**2)
    H = 1.0/(1.0+D**(2*order))
    return (1-H) if high_pass else H

def fft_hp(img, cutoff=0.1, order=2):
    F = fftshift(fft2(img))
    return normalize(np.abs(ifft2(ifftshift(F*_butterworth(img.shape, cutoff, order, True)))))

def fft_lp(img, cutoff=0.15, order=2):
    F = fftshift(fft2(img))
    return normalize(np.abs(ifft2(ifftshift(F*_butterworth(img.shape, cutoff, order, False)))))

def fft_spectrum(img):
    return np.log1p(np.abs(fftshift(fft2(img))))


# ══════════════════════════════════════════════════════════════
#  SUPPRESSION D'ARRIÈRE-PLAN — ALGORITHMES AMÉLIORÉS
# ══════════════════════════════════════════════════════════════

# ── Saillance spectrale (Hou & Zhang 2007) ────────────────────

def spectral_saliency(img_gray):
    """
    Carte de saillance par résidu spectral.
    Identifie les régions visuellement saillantes (probable premier plan).
    """
    f    = np.fft.fft2(img_gray.astype(np.float32))
    log_amp = np.log(np.abs(f) + 1e-6)
    # Résidu = log amplitude - version lissée
    residual = log_amp - cv2.blur(log_amp, (3, 3))
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * np.angle(f))))**2
    sal = cv2.GaussianBlur(sal.real.astype(np.float32), (11, 11), 0)
    return cv2.normalize(sal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# ── GrabCut v2 avec initialisation par saillance ─────────────

def bg_remove_grabcut(img_color, n_iter=10, margin_ratio=0.04):
    """
    GrabCut amélioré : initialisation du masque par saillance spectrale.
    Zones saillantes → probable premier plan.
    Zones de faible saillance + bordures → probable fond.
    """
    h, w   = img_color.shape[:2]
    margin = max(2, int(min(h, w) * margin_ratio))
    gray   = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY).astype(np.float64)/255.0
    sal    = spectral_saliency(gray)

    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)   # défaut : prob. fond

    # Haute saillance → prob. premier plan
    _, sal_med = cv2.threshold(sal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask[sal_med > 0] = cv2.GC_PR_FGD

    # Très haute saillance → définitivement premier plan
    _, sal_hi = cv2.threshold(sal, int(sal.max()*0.75), 255, cv2.THRESH_BINARY)
    mask[sal_hi > 0] = cv2.GC_FGD

    # Bordures → définitivement fond
    mask[:margin, :]  = cv2.GC_BGD
    mask[-margin:, :] = cv2.GC_BGD
    mask[:, :margin]  = cv2.GC_BGD
    mask[:, -margin:] = cv2.GC_BGD

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_color, mask, None, bgd, fgd, n_iter, cv2.GC_INIT_WITH_MASK)
    except Exception:
        rect = (margin, margin, w-2*margin, h-2*margin)
        mask2 = np.zeros((h, w), np.uint8)
        cv2.grabCut(img_color, mask2, rect, bgd, fgd, n_iter, cv2.GC_INIT_WITH_RECT)
        return np.where((mask2==2)|(mask2==0), 0, 1).astype(np.uint8)

    return np.where((mask==2)|(mask==0), 0, 1).astype(np.uint8)


# ── Flood Fill depuis les bordures ───────────────────────────

def bg_remove_border_flood(img_color, tolerance=25):
    """
    Fond = toutes les régions connectées aux bords de l'image.
    Très efficace pour photos sur fond blanc, uni ou gradué.
    Travaille en espace LAB pour meilleure robustesse colorimétrique.
    """
    h, w   = img_color.shape[:2]
    lab    = cv2.cvtColor(img_color, cv2.COLOR_BGR2LAB)
    gray   = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    flood_mask = np.zeros((h+2, w+2), np.uint8)
    work = blurred.copy()

    # Flood fill depuis tous les pixels des 4 bords (pas seulement les coins)
    border_pts = (
        [(x, 0)   for x in range(0, w, 3)] +
        [(x, h-1) for x in range(0, w, 3)] +
        [(0, y)   for y in range(0, h, 3)] +
        [(w-1, y) for y in range(0, h, 3)]
    )
    for (x, y) in border_pts:
        if flood_mask[y+1, x+1] == 0:
            cv2.floodFill(work, flood_mask, (x, y), 128,
                         loDiff=tolerance, upDiff=tolerance)

    bg = (flood_mask[1:-1, 1:-1] > 0).astype(np.uint8)
    return 1 - bg


# ── K-means en espace LAB + position ────────────────────────

def bg_remove_kmeans(img_color, k=3):
    """
    Segmentation K-means dans l'espace LAB (perceptuellement uniforme)
    augmenté des coordonnées normalisées (Y, X).
    Le cluster dominant aux bords = fond.
    """
    h, w   = img_color.shape[:2]
    lab    = cv2.cvtColor(img_color, cv2.COLOR_BGR2LAB).astype(np.float32)/255.0
    Y, X   = np.mgrid[0:h, 0:w]
    Yn, Xn = Y/h, X/w

    # 5 features : L, A, B + position (poids réduit)
    feats = np.column_stack([
        lab[:,:,0].ravel(),
        lab[:,:,1].ravel(),
        lab[:,:,2].ravel(),
        Yn.ravel() * 0.25,
        Xn.ravel() * 0.25,
    ]).astype(np.float32)

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, _ = cv2.kmeans(feats, k, None, crit, 10, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(h, w)

    # Fond = cluster le plus présent sur les bordures
    border = np.zeros((h, w), bool)
    b = 6
    border[:b,:] = border[-b:,:] = border[:,:b] = border[:,-b:] = True
    counts = [np.sum(labels[border]==i) for i in range(k)]
    bg_lbl = np.argmax(counts)

    # Si k=3 : les 2 clusters les moins saillants aux bords sont fond
    fg_mask = (labels != bg_lbl).astype(np.uint8)
    if k == 3:
        sorted_bg = np.argsort(counts)[::-1]
        for extra_bg in sorted_bg[1:2]:
            # Exclure si trop présent au centre
            cy, cx = h//2, w//2
            center = labels[cy-h//6:cy+h//6, cx-w//6:cx+w//6]
            if np.sum(center==extra_bg) < center.size * 0.3:
                fg_mask[labels==extra_bg] = 0

    return fg_mask


# ── rembg (IA — U²-Net) ──────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_rembg_session(model_name):
    return new_session(model_name)

def bg_remove_rembg(img_rgb, model_name="u2net", do_alpha=False, af=240, ab=10, ae=10, post_process=False):
    pil  = Image.fromarray(img_rgb)
    session = get_rembg_session(model_name)
    out  = rembg_remove(
        pil,
        session=session,
        alpha_matting=do_alpha,
        alpha_matting_foreground_threshold=af,
        alpha_matting_background_threshold=ab,
        alpha_matting_erode_size=ae,
        post_process_mask=post_process,
        only_mask=True
    )
    mask = np.array(out)
    return (mask > 128).astype(np.uint8)


# ══════════════════════════════════════════════════════════════
#  RAFFINEMENT DU MASQUE
# ══════════════════════════════════════════════════════════════

def fill_holes(mask):
    """Remplit les trous internes (régions fond entourées de sujet)."""
    inv = (1 - mask).astype(np.uint8)
    n, labels = cv2.connectedComponents(inv)
    # Composantes ne touchant pas la bordure = trous
    border_lbls = set()
    for row in [labels[0,:], labels[-1,:]]:
        border_lbls.update(np.unique(row))
    for col in [labels[:,0], labels[:,-1]]:
        border_lbls.update(np.unique(col))
    holes = np.zeros_like(mask)
    for i in range(1, n):
        if i not in border_lbls:
            holes[labels==i] = 1
    return np.clip(mask + holes, 0, 1).astype(np.uint8)

def remove_islands(mask, min_ratio=0.003):
    """Supprime les petites régions isolées (bruit de segmentation)."""
    min_area = mask.size * min_ratio
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels==i] = 1
    return clean

def edge_snap(mask, img_gray, snap_radius=4):
    """
    Aligne les bords du masque sur les contours Canny les plus proches.
    Évite les bords flous ou décalés.
    """
    edges  = cv2.Canny(to_uint8(img_gray), 40, 120)
    # Dilatation des contours pour créer une zone d'attraction
    k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (snap_radius*2+1,)*2)
    edge_z = cv2.dilate(edges, k)
    # Éroder légèrement le masque pour le faire rentrer dans les contours
    k_sm   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(mask, k_sm)
    # Là où un bord est proche, utiliser le masque érodé (bord plus serré)
    result = mask.copy()
    boundary = mask - eroded                    # anneau = bord du masque
    result[boundary==1] = np.where(edge_z[boundary==1]>0, 0, 1)
    return result

def alpha_matting(mask, radius=8):
    """
    Alpha matting par distance transform.
    Crée une transition douce fg → bg pour des bords naturels.
    """
    dist_fg = cv2.distanceTransform(mask,   cv2.DIST_L2, 5)
    dist_bg = cv2.distanceTransform(1-mask, cv2.DIST_L2, 5)
    alpha   = mask.copy().astype(np.float32)
    # Zone de transition à l'intérieur du masque
    inner = (dist_fg < radius) & (mask == 1)
    alpha[inner] = dist_fg[inner] / radius
    # Zone de transition à l'extérieur
    outer = (dist_bg < radius) & (mask == 0)
    alpha[outer] = 1.0 - dist_bg[outer] / radius
    return np.clip(alpha, 0, 1)

def full_refinement(mask, img_gray, do_fill=True, do_islands=True,
                    do_snap=True, snap_r=4, do_morph=True, alpha_r=6):
    """Pipeline complet de raffinement du masque."""
    m = mask.copy()
    if do_morph:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    if do_fill:
        m = fill_holes(m)
    if do_islands:
        m = remove_islands(m)
    if do_snap:
        m = edge_snap(m, img_gray, snap_r)
    alpha = alpha_matting(m, alpha_r)
    return m, alpha


# ══════════════════════════════════════════════════════════════
#  APPLICATION DU MASQUE
# ══════════════════════════════════════════════════════════════

def apply_mask_soft(img_rgb, alpha, bg_color=(0,0,0)):
    """Application du masque avec alpha doux (feathering)."""
    a3   = alpha[:,:,np.newaxis]
    bg   = np.array(bg_color, dtype=np.float32)
    out  = img_rgb.astype(np.float32) * a3 + bg * (1 - a3)
    return np.clip(out, 0, 255).astype(np.uint8)

def to_rgba(img_rgb, alpha):
    a8   = (alpha * 255).astype(np.uint8)
    return np.dstack([img_rgb, a8])

def color_transfer(source, target):
    """
    Transfère les statistiques de couleur (moyenne, écart-type) de target vers source
    en utilisant l'espace LAB (Reinhard et al. 2001).
    """
    s_lab = cv2.cvtColor(source.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    t_lab = cv2.cvtColor(target.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)

    s_mean, s_std = cv2.meanStdDev(s_lab)
    t_mean, t_std = cv2.meanStdDev(t_lab)

    s_mean = s_mean.flatten()
    s_std  = s_std.flatten()
    t_mean = t_mean.flatten()
    t_std  = t_std.flatten()

    for i in range(3):
        # Éviter division par zéro
        if s_std[i] < 1e-6: continue
        s_lab[:,:,i] = ((s_lab[:,:,i] - s_mean[i]) * (t_std[i] / s_std[i])) + t_mean[i]

    res = cv2.cvtColor(np.clip(s_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return res

def transform_patch(img, mask, scale=1.0, angle=0.0):
    """Redimensionne et fait pivoter une image et son masque."""
    h, w = img.shape[:2]
    # Redimensionnement
    if scale != 1.0:
        nh, nw = int(h * scale), int(w * scale)
        if nh > 0 and nw > 0:
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
            mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
            h, w = nh, nw

    # Rotation
    if angle != 0:
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Calculer la taille de la boîte englobante pour éviter de couper les bords
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]
        
        img = cv2.warpAffine(img, matrix, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        mask = cv2.warpAffine(mask, matrix, (new_w, new_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return img, mask

def apply_reflection(img_rgb, alpha, opacity=0.3, length=0.4, gap=2):
    """
    Ajoute un reflet miroir dégradé sous le sujet.
    img_rgb: Image avec fond déjà appliqué ou transparent.
    alpha: Masque de l'objet uniquement.
    """
    h, w = img_rgb.shape[:2]
    # Trouver les limites du sujet
    y_idx, x_idx = np.where(alpha > 0.1)
    if len(y_idx) == 0: return img_rgb
    
    y_max = y_idx.max()
    
    # Créer le miroir
    reflection_h = int(h * length)
    if reflection_h <= 0: return img_rgb
    
    # On inverse l'image
    flipped_img = cv2.flip(img_rgb, 0)
    flipped_alpha = cv2.flip(alpha, 0)
    
    # On calcule la position de départ dans l'image inversée
    # Le bas de l'objet original devient le haut de l'objet inversé
    start_y = h - y_max - 1
    
    res = img_rgb.copy().astype(np.float32)
    
    # Appliquer le reflet ligne par ligne avec un dégradé
    for i in range(reflection_h):
        target_y = y_max + gap + i
        source_y = start_y + i
        
        if target_y >= h or source_y >= h: break
        
        # Dégradé linéaire
        grad = opacity * (1.0 - i / reflection_h)
        
        mask = flipped_alpha[source_y, :] * grad
        mask_3d = np.dstack([mask]*3)
        
        # Mélange
        res[target_y, :] = res[target_y, :] * (1 - mask_3d) + flipped_img[source_y, :] * mask_3d
        
    return np.clip(res, 0, 255).astype(np.uint8)

def auto_crop_to_subject(img, alpha, margin=20):
    """Recadre l'image au plus près du sujet détecté."""
    y_idx, x_idx = np.where(alpha > 0.05)
    if len(y_idx) == 0: return img
    h, w = img.shape[:2]
    y1, y2 = max(0, y_idx.min() - margin), min(h, y_idx.max() + margin)
    x1, x2 = max(0, x_idx.min() - margin), min(w, x_idx.max() + margin)
    return img[y1:y2, x1:x2]

def apply_image_adjustments(img, brightness=0, contrast=0, saturation=1.0, sharpness=0):
    """Ajustements finaux de l'image (Luminosité, Contraste, Saturation, Netteté)."""
    # Luminosité et Contraste
    alpha_c = 1.0 + (contrast / 100.0)
    res = cv2.convertScaleAbs(img, alpha=alpha_c, beta=brightness)
    
    # Saturation
    if saturation != 1.0:
        hsv = cv2.cvtColor(res, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:,:,1] *= saturation
        hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
        res = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
    # Netteté (Unsharp Masking)
    if sharpness > 0:
        blur = cv2.GaussianBlur(res, (0, 0), 3)
        res = cv2.addWeighted(res, 1.0 + sharpness, blur, -sharpness, 0)
        
    return res

def apply_artistic_mode(img_rgb, mode="Sketch"):
    """Transforme l'image en œuvre d'art (Croquis ou Blueprint)."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    if mode == "Croquis (Sketch)":
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
    elif mode == "Bleu technique (Blueprint)":
        edges = cv2.Canny(gray, 30, 100)
        # Fond bleu profond
        blueprint = np.zeros_like(img_rgb)
        blueprint[:] = (20, 60, 120) 
        # Lignes blanches
        blueprint[edges > 0] = (255, 255, 255)
        # Optionnel: grillage léger
        h, w = blueprint.shape[:2]
        blueprint[0:h:40, :] = blueprint[0:h:40, :] * 0.8 + np.array([50, 100, 180])*0.2
        blueprint[:, 0:w:40] = blueprint[:, 0:w:40] * 0.8 + np.array([50, 100, 180])*0.2
        return blueprint
    return img_rgb

def generate_shadow(mask, blur=15, opacity=0.5, dx=10, dy=10):
    """Génère une couche d'ombre à partir d'un masque alpha."""
    # Créer une image noire de la taille du masque
    h, w = mask.shape
    # Appliquer le décalage
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted_mask = cv2.warpAffine(mask.astype(np.float32), M, (w, h))
    # Flou
    if blur > 0:
        if blur % 2 == 0: blur += 1
        shifted_mask = cv2.GaussianBlur(shifted_mask, (blur, blur), 0)
    
    return shifted_mask * opacity

def apply_watermark(img, text, color=(255, 255, 255), opacity=0.3, scale=1.2, position="Bas-Droite", use_outline=True):
    """
    Ajoute un texte de filigrane sur l'image avec contour pour la visibilité.
    Supporte les couleurs personnalisées et le positionnement robuste.
    """
    if not text: return img
    
    # S'assurer que l'image est en uint8 pour OpenCV
    if img.dtype != np.uint8:
        res = to_uint8(img)
    else:
        res = img.copy()
        
    h, w = res.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    
    # Calculer l'épaisseur en fonction de l'échelle
    thickness = max(1, int(1.5 * scale))
    outline_thickness = thickness + max(1, int(2 * scale))
    
    # Obtenir la taille du texte
    size, baseline = cv2.getTextSize(text, font, scale, thickness)
    tw, th = size
    
    # Marges sécurisées
    margin = int(min(w, h) * 0.02) # 2% de marge
    
    if position == "Bas-Droite":
        px, py = w - tw - margin, h - margin
    elif position == "Bas-Gauche":
        px, py = margin, h - margin
    elif position == "Haut-Droite":
        px, py = w - tw - margin, th + margin
    elif position == "Haut-Gauche":
        px, py = margin, th + margin
    elif position == "Centre":
        px, py = (w - tw) // 2, (h + th) // 2
    else: # Par défaut Bas-Droite
        px, py = w - tw - margin, h - margin

    # S'assurer que les coordonnées restent dans l'image
    px = max(0, min(px, w - tw))
    py = max(th, min(py, h))

    overlay = res.copy()
    
    # OpenCV utilise BGR, mais nos images Streamlit sont RGB
    # On inverse la couleur si nécessaire ou on l'utilise telle quelle selon le contexte
    # Ici on suppose que l'image passée est déjà en RGB
    
    if use_outline:
        # Contour noir (ou blanc si le texte est très sombre pour garder du contraste)
        outline_color = (0, 0, 0) if sum(color) > 100 else (255, 255, 255)
        cv2.putText(overlay, text, (px, py), font, scale, outline_color, outline_thickness, cv2.LINE_AA)
        
    # Texte principal
    cv2.putText(overlay, text, (px, py), font, scale, color, thickness, cv2.LINE_AA)
    
    # Fusionner avec l'image originale pour l'opacité
    cv2.addWeighted(overlay, opacity, res, 1 - opacity, 0, res)
    return res

# ══════════════════════════════════════════════════════════════
#  PIPELINE CONTOURS
# ══════════════════════════════════════════════════════════════

def compute_all(img, p):
    enhanced = clahe_enhance(img)
    smooth_g = gaussian_smoothing(enhanced, p["sigma"])
    return {
        "🖼 Original":             (img,                                               "gray"),
        "✨ CLAHE":                 (enhanced,                                         "gray"),
        "🔵 Gaussien":             (smooth_g,                                         "gray"),
        "🟠 Bilatéral":            (bilateral_smoothing(enhanced,p["bil_c"],p["bil_s"]),"gray"),
        "🟡 Médian":               (median_smoothing(enhanced, p["med_k"]),            "gray"),
        "🟢 Spline Lissée":        (spline_smooth(enhanced, p["spl_scale"]),           "gray"),
        "🔴 Sobel + NMS":          (sobel_nms(smooth_g),                              "hot"),
        "⚡ Canny":                 (canny_edges(enhanced, p["clo"], p["chi"]),        "gray"),
        "🔮 LoG Zéro-croisements": (log_zero_crossing(enhanced, p["sigma"]),          "hot"),
        "🌿 Contours Spline":      (spline_edges(enhanced, p["spl_scale"]),           "hot"),
        "🟣 FFT Spectre":          (fft_spectrum(img),                                "inferno"),
        "📈 FFT Passe-haut":       (fft_hp(enhanced,p["fft_cut"],p["fft_ord"]),       "gray"),
        "📉 FFT Passe-bas":        (fft_lp(enhanced,p["fft_cut"],p["fft_ord"]),       "gray"),
    }


# ══════════════════════════════════════════════════════════════
#  POISSON IMAGE BLENDING (SEAMLESS CLONING)
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def get_poisson_operators(m, n):
    Dx = sc.lil_matrix((m*n, m*n))
    Dx.setdiag(-1)
    I = sc.lil_matrix((n, n))
    I.setdiag(1)
    for k in range(m-1):
        Dx[k*n:k*n+n, k*n+n:k*n+2*n] = I
    Dx = Dx.tocsc()

    Dy = sc.lil_matrix((m*n, m*n))
    Dy.setdiag(-1)
    Dy.setdiag(1, 1)
    Dy = Dy.tocsc()

    D = sc.lil_matrix((n, n))
    D.setdiag(-4)
    D.setdiag(1, 1)
    D.setdiag(1, -1)
    I2 = sc.lil_matrix((n, n))
    I2.setdiag(1)
    A = sc.lil_matrix((m*n, m*n))
    for k in range(m):
        A[k*n:k*n+n, k*n:k*n+n] = D
    for k in range(m-1):
        A[k*n:k*n+n, k*n+n:k*n+2*n] = I2
    for k in range(m-1):
        A[k*n+n:k*n+2*n, k*n:k*n+n] = I2
    A = A.tocsc()
    return Dx, Dy, A

def make_S_sparse(mask):
    m, n = mask.shape
    N1 = int(np.sum(mask))
    N = m * n
    indices = np.where(mask.flatten() == 1)[0]
    S = sc.csr_matrix((np.ones(N1), (np.arange(N1), indices)), shape=(N1, N))
    return S

def poisson_blend_patch(source, target, mask, mode="Normal"):
    m, n = mask.shape
    S = make_S_sparse(mask)
    Dx, Dy, A = get_poisson_operators(m, n)
    
    A_final = S.dot(A.dot(S.T))
    result = np.zeros_like(target)
    
    for c in range(3):
        s = source[:,:,c].flatten().astype(np.float64)
        t = target[:,:,c].flatten().astype(np.float64)
        
        r = t.copy()
        r[mask.flatten() == 1] = 0
        
        if mode == "Normal":
            b = S.dot(A.dot(s)) - S.dot(A.dot(r))
        else:
            vx = Dx.dot(s); vy = Dy.dot(s)
            wx = Dx.dot(t); wy = Dy.dot(t)
            v_norm = vx**2 + vy**2
            w_norm = wx**2 + wy**2
            
            if mode == "Gradient Max":
                qx = np.where(v_norm > w_norm, vx, wx)
                qy = np.where(v_norm > w_norm, vy, wy)
            else: # Gradient Min
                qx = np.where(v_norm < w_norm, vx, wx)
                qy = np.where(v_norm < w_norm, vy, wy)
                
            b = S.dot(Dx.dot(qx) + Dy.dot(qy) - A.dot(r))
            
        u = spsolve(A_final, b)
        
        x = t.copy()
        x[mask.flatten() == 1] = u
        result[:,:,c] = np.clip(x, 0, 255).reshape((m, n))
        
    return result


@st.cache_data(show_spinner="⏳ Calcul contours…")
def cached_contours(img_bytes, pt):
    arr  = np.frombuffer(img_bytes, np.uint8)
    img_ = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE).astype(np.float64)/255.0
    return compute_all(img_, dict(pt))

@st.cache_data(show_spinner="⏳ Segmentation en cours…")
def cached_bg(img_bytes, method, gc_iter, gc_margin, ff_tol, km_k,
              rembg_model, rembg_post_process, rembg_alpha, rembg_af, rembg_ab, rembg_ae,
              do_fill, do_islands, do_snap, snap_r, alpha_r):
    arr   = np.frombuffer(img_bytes, np.uint8)
    color = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    rgb   = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    gray_ = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY).astype(np.float64)/255.0

    if   "GrabCut"  in method: raw_mask = bg_remove_grabcut(color, gc_iter, gc_margin/max(color.shape[:2]))
    elif "Flood"    in method: raw_mask = bg_remove_border_flood(color, ff_tol)
    elif "K-means"  in method: raw_mask = bg_remove_kmeans(color, km_k)
    elif "rembg"    in method: raw_mask = bg_remove_rembg(rgb, rembg_model, rembg_alpha, rembg_af, rembg_ab, rembg_ae, rembg_post_process)
    else:                      raw_mask = np.ones(gray_.shape, np.uint8)

    refined_mask, alpha = full_refinement(
        raw_mask, gray_, do_fill, do_islands, do_snap, snap_r, True, alpha_r)
    sal = spectral_saliency(gray_)
    return raw_mask, refined_mask, alpha, rgb, gray_, sal


# ══════════════════════════════════════════════════════════════
#  INTERFACE STREAMLIT
# ══════════════════════════════════════════════════════════════

st.set_page_config(page_title="Détection de Contours", page_icon="🔬", layout="wide")

st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background: #2a2a3e;
        overflow-y: auto !important; overflow-x: hidden !important;
        max-height: 100vh;
    }
    [data-testid="stSidebar"] > div:first-child {
        overflow-y: auto !important; height: 100vh; padding-bottom: 2rem;
    }
    [data-testid="stSidebarContent"] {
        overflow-y: auto !important; height: 100% !important; padding-bottom: 2rem !important;
    }
    [data-testid="stSidebar"] * { color: #cdd6f4 !important; }
    h1,h2,h3 { color: #cba6f7 !important; }
    .stSelectbox label, .stSlider label { color: #89b4fa !important; font-weight:600; }
</style>
""", unsafe_allow_html=True)

st.title("🔬 Détection, Lissage & Suppression d'Arrière-plan")
rembg_badge = " · **rembg IA** ✅" if REMBG_AVAILABLE else " · rembg _(pip install rembg)_"
st.caption("Sobel·NMS · Canny · LoG · Bilatéral · FFT·Butterworth · Splines · "
           "GrabCut+Saillance · FloodFill · K-means LAB" + rembg_badge)
st.divider()

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Paramètres")
    uploaded_files = st.file_uploader("📁 Image(s)", type=["png","jpg","jpeg","bmp","tiff","webp"], accept_multiple_files=True)
    if uploaded_files:
        if len(uploaded_files) > 1:
            img_names = [f.name for f in uploaded_files]
            selected_img_name = st.selectbox("🖼 Image à traiter", img_names)
            uploaded = next(f for f in uploaded_files if f.name == selected_img_name)
        else:
            uploaded = uploaded_files[0]
    else:
        uploaded = None

    st.subheader("🔵 Gaussien / LoG")
    sigma    = st.slider("σ sigma",     0.3, 6.0, 1.5, 0.1)

    st.subheader("⚡ Canny")
    clo = st.slider("Seuil bas",  0.01, 0.30, 0.05, 0.01)
    chi = st.slider("Seuil haut", 0.05, 0.60, 0.18, 0.01)

    st.subheader("🟠 Bilatéral")
    bil_c = st.slider("σ couleur", 0.01, 0.50, 0.10, 0.01)
    bil_s = st.slider("σ spatial",  2.0, 30.0, 10.0, 0.5)

    st.subheader("🟡 Médian")
    med_k = st.slider("Taille noyau", 3, 15, 5, 2)

    st.subheader("🟣 FFT Butterworth")
    fft_cut = st.slider("Cutoff", 0.02, 0.40, 0.08, 0.01)
    fft_ord = st.slider("Ordre",     1,    6,    2,    1)

    st.subheader("🟢 Spline")
    spl_scale = st.slider("Échelle", 0.1, 1.0, 0.5, 0.05)

    st.subheader("✂️ Suppression Arrière-plan")
    bg_opts = []
    if REMBG_AVAILABLE:
        bg_opts.append("rembg (IA U²-Net)")
    bg_opts.extend(["GrabCut + Saillance", "Flood Fill Bordures", "K-means LAB"])
    
    bg_method  = st.selectbox("Méthode", bg_opts)
    gc_iter    = st.slider("GrabCut — itérations",  3, 15, 10) if "GrabCut" in bg_method else 10
    gc_margin  = st.slider("GrabCut — marge",       1, 80, 15) if "GrabCut" in bg_method else 15
    ff_tol     = st.slider("FloodFill — tolérance", 5, 80, 25) if "Flood" in bg_method else 25
    km_k       = st.slider("K-means — clusters",    2,  5,  3) if "K-means" in bg_method else 3

    if "rembg" in bg_method:
        st.markdown("**⚙️ Paramètres rembg**")
        rembg_model = st.selectbox("Modèle IA", [
            "u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta",
            "isnet-general-use", "isnet-anime", 
            "birefnet-general", "birefnet-general-lite", "birefnet-portrait", 
            "birefnet-dis", "birefnet-hrsod", "birefnet-cod", "birefnet-massive",
            "sam", "ben2-base"
        ])
        rembg_post_process = st.checkbox("Post-traitement du masque (rembg)", value=False)
        rembg_alpha = st.checkbox("Alpha Matting (rembg)", value=False)
        if rembg_alpha:
            rembg_af = st.slider("Seuil premier plan (rembg)", 0, 255, 240)
            rembg_ab = st.slider("Seuil arrière-plan (rembg)", 0, 255, 10)
            rembg_ae = st.slider("Érosion (rembg)", 1, 50, 10)
        else:
            rembg_af = 240; rembg_ab = 10; rembg_ae = 10
    else:
        rembg_model = "u2net"
        rembg_post_process = False
        rembg_alpha = False
        rembg_af = 240; rembg_ab = 10; rembg_ae = 10

    st.markdown("**Raffinement du masque**")
    do_fill    = st.checkbox("Remplir les trous",       value=True)
    do_islands = st.checkbox("Supprimer îlots isolés",  value=True)
    do_snap    = st.checkbox("Edge snapping (Canny)",   value=True)
    snap_r     = st.slider("Rayon snap (px)", 1, 12, 4) if do_snap else 4
    alpha_r    = st.slider("Alpha matting (px)", 0, 20, 7)

    bg_color_hex = st.color_picker("Couleur du fond", "#ffffff")
    hx = bg_color_hex.lstrip("#")
    bg_rgb_tuple = tuple(int(hx[i:i+2], 16) for i in (0, 2, 4))

    st.divider()
    st.subheader("🐟 Incrustation Poisson")
    target_uploaded = st.file_uploader("🖼 Image Cible (Fond)", type=["png","jpg","jpeg","bmp","tiff","webp"])

    st.divider()
    st.subheader("🖋️ Filigrane (Watermark)")
    wm_text = st.text_input("Texte", "")
    c_wm1, c_wm2 = st.columns(2)
    wm_color_hex = c_wm1.color_picker("Couleur", "#ffffff")
    wm_op   = c_wm2.slider("Opacité", 0.0, 1.0, 0.6)
    
    # Conversion hex en RGB
    wm_color_rgb = tuple(int(wm_color_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    
    wm_size = st.slider("Taille texte", 0.5, 5.0, 1.5, 0.1)
    wm_pos  = st.selectbox("Position", ["Bas-Droite", "Bas-Gauche", "Haut-Gauche", "Haut-Droite", "Centre"])
    wm_outline = st.checkbox("Contour de visibilité", value=True)
    wm_on_all = st.checkbox("Afficher sur toutes les vues", value=False)

    if uploaded_files and len(uploaded_files) > 1:
        st.divider()
        st.subheader("📦 Traitement par lot")
        
        # Hachage des paramètres pour détecter tout changement et réinitialiser le ZIP
        import hashlib
        param_str = f"{bg_method}-{gc_iter}-{gc_margin}-{ff_tol}-{km_k}-{rembg_model}-{rembg_post_process}-{rembg_alpha}-{rembg_af}-{rembg_ab}-{rembg_ae}-{do_fill}-{do_islands}-{do_snap}-{snap_r}-{alpha_r}-{wm_text}-{wm_color_rgb}-{wm_op}-{wm_size}-{wm_pos}-{wm_outline}-{len(uploaded_files)}-" + "-".join([f"{f.name}_{f.size}" for f in uploaded_files])
        param_hash = hashlib.md5(param_str.encode()).hexdigest()
        
        if st.session_state.get("last_param_hash") != param_hash:
            st.session_state["zip_data"] = None
            st.session_state["last_param_hash"] = param_hash
            
        if st.button("Générer ZIP (fond supprimé)", use_container_width=True):
            zip_buffer = io.BytesIO()
            with st.spinner("Traitement du lot en cours..."):
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for f in uploaded_files:
                        f_bytes = f.getvalue()
                        _, _, f_alpha, f_rgb, _, _ = cached_bg(
                            f_bytes, bg_method, gc_iter, gc_margin, ff_tol, km_k,
                            rembg_model, rembg_post_process, rembg_alpha, rembg_af, rembg_ab, rembg_ae,
                            do_fill, do_islands, do_snap, snap_r, alpha_r)
                        
                        f_rgba = to_rgba(f_rgb, f_alpha)
                        if wm_text:
                            rgb_wm = apply_watermark(f_rgba[:,:,:3], wm_text, wm_color_rgb, wm_op, wm_size, wm_pos, wm_outline)
                            f_rgba = np.dstack([rgb_wm, f_rgba[:,:,3]])
                        
                        img_pil = Image.fromarray(f_rgba)
                        buf = io.BytesIO()
                        img_pil.save(buf, format="PNG")
                        zf.writestr(f.name.split('.')[0] + "_no_bg.png", buf.getvalue())
            st.session_state["zip_data"] = zip_buffer.getvalue()
            st.success("✅ ZIP généré avec succès ! Cliquez sur Télécharger ci-dessous.")
            
        if st.session_state.get("zip_data") is not None:
            st.download_button("💾 Télécharger le ZIP", st.session_state["zip_data"], "batch_export.zip", "application/zip", use_container_width=True)

    st.divider()
    st.subheader("🌟 Fonctions Avancées")
    do_crop = st.checkbox("Recadrage auto au sujet", value=False)
    do_reflect = st.checkbox("Ajouter un reflet miroir", value=False)
    if do_reflect:
        ref_op = st.slider("Opacité reflet", 0.0, 1.0, 0.3)
        ref_len = st.slider("Longueur reflet", 0.1, 0.8, 0.3)
    
    st.markdown("**🎨 Retouches & Filtres**")
    art_mode = st.selectbox("Filtre artistique", ["Aucun", "Croquis (Sketch)", "Bleu technique (Blueprint)"])
    
    c_adj1, c_adj2 = st.columns(2)
    adj_bright = c_adj1.slider("Luminosité", -100, 100, 0)
    adj_cont   = c_adj2.slider("Contraste", -100, 100, 0)
    adj_sat    = c_adj1.slider("Saturation", 0.0, 2.0, 1.0, 0.1)
    adj_sharp  = c_adj2.slider("Netteté", 0.0, 2.0, 0.0, 0.1)

    st.divider()
    st.subheader("🎨 Colormap")
    custom_cmap = st.selectbox("Colormap", ["Auto","gray","hot","inferno","viridis","plasma","coolwarm"])

    st.button("▶ Appliquer", type="primary", use_container_width=True)

# ── Chargement ───────────────────────────────────────────────
if uploaded is None:
    st.info("👈 Chargez une image dans le panneau gauche.", icon="📂")
    st.stop()

raw_bytes = np.frombuffer(uploaded.read(), np.uint8)
raw_gray  = cv2.imdecode(raw_bytes, cv2.IMREAD_GRAYSCALE)
raw_color = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
raw_rgb   = cv2.cvtColor(raw_color, cv2.COLOR_BGR2RGB)
if raw_gray is None:
    st.error("Impossible de lire l'image."); st.stop()

img = raw_gray.astype(np.float64)/255.0
h, w = img.shape

c1,c2,c3 = st.columns(3)
c1.metric("Largeur", f"{w} px"); c2.metric("Hauteur", f"{h} px"); c3.metric("Taille", f"{w*h//1000} Kpx")
st.divider()

params = dict(sigma=sigma, clo=clo, chi=chi, bil_c=bil_c, bil_s=bil_s,
              med_k=med_k, fft_cut=fft_cut, fft_ord=fft_ord, spl_scale=spl_scale)

results = cached_contours(uploaded.getvalue(), tuple(sorted(params.items())))
raw_mask, refined_mask, alpha, orig_rgb, orig_gray, sal_map = cached_bg(
    uploaded.getvalue(), bg_method, gc_iter, gc_margin, ff_tol, km_k,
    rembg_model, rembg_post_process, rembg_alpha, rembg_af, rembg_ab, rembg_ae,
    do_fill, do_islands, do_snap, snap_r, alpha_r)

method_names = list(results.keys())

# ── Onglets ──────────────────────────────────────────────────
tab_single, tab_all, tab_compare, tab_bg, tab_poisson, tab_wipe = st.tabs([
    "🔍 Vue unique", "🖼 Vue complète", "⚖️ Comparaison", "✂️ Suppression Fond", "🎨 Composition & Poisson", "🌓 Avant/Après"
])

# Tab 1
with tab_single:
    sel = st.selectbox("Méthode", method_names, index=6)
    data, ac = results[sel]
    if wm_on_all and wm_text:
        # Convertir en RGB si c'est du grayscale avant d'ajouter le filigrane
        if len(data.shape) == 2:
            import matplotlib.cm as cm
            cmap_obj = cm.get_cmap(ac if custom_cmap=="Auto" else custom_cmap)
            data_rgb = (cmap_obj(data)[:,:,:3] * 255).astype(np.uint8)
        else:
            data_rgb = to_uint8(data)
        data_to_show = apply_watermark(data_rgb, wm_text, wm_color_rgb, wm_op, wm_size, wm_pos, wm_outline)
        is_rgb = True
    else:
        data_to_show = data
        is_rgb = False
        
    cmap = ac if custom_cmap=="Auto" else custom_cmap
    fig, ax = plt.subplots(figsize=(8,6), facecolor="#1e1e2e")
    ax.set_facecolor("#1e1e2e")
    if is_rgb:
        im = ax.imshow(data_to_show, interpolation="nearest")
    else:
        im = ax.imshow(data_to_show, cmap=cmap, interpolation="nearest")
    ax.set_title(sel, color="#cdd6f4", fontsize=14, fontweight="bold", pad=12)
    ax.axis("off")
    if not is_rgb:
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
    buf = io.BytesIO()
    f2,a2 = plt.subplots(figsize=(8,6))
    if is_rgb:
        a2.imshow(data_to_show)
    else:
        a2.imshow(data_to_show, cmap=cmap)
    a2.axis("off")
    f2.tight_layout(); f2.savefig(buf, format="png", dpi=150, bbox_inches="tight"); plt.close(f2)
    st.download_button("💾 Télécharger", buf.getvalue(),
                       file_name=f"contour_{sel.split()[-1].lower()}.png", mime="image/png")

# Tab 2
with tab_all:
    ncols=4; nrows=(len(results)+ncols-1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows*4), facecolor="#1e1e2e")
    for i,(name,(data,ac)) in enumerate(results.items()):
        cmap = ac if custom_cmap=="Auto" else custom_cmap
        if wm_on_all and wm_text:
            if len(data.shape) == 2:
                import matplotlib.cm as cm
                cmap_obj = cm.get_cmap(cmap)
                data_rgb = (cmap_obj(data)[:,:,:3] * 255).astype(np.uint8)
            else:
                data_rgb = to_uint8(data)
            grid_wm_size = wm_size * 0.6
            data_to_show = apply_watermark(data_rgb, wm_text, wm_color_rgb, wm_op, grid_wm_size, wm_pos, wm_outline)
            axes.flat[i].imshow(data_to_show, interpolation="nearest")
        else:
            axes.flat[i].imshow(data, cmap=cmap, interpolation="nearest")
        axes.flat[i].set_title(name, color="#cdd6f4", fontsize=9, fontweight="bold")
        axes.flat[i].axis("off")
    for j in range(i+1, nrows*ncols): axes.flat[j].axis("off")
    fig.suptitle("Toutes les méthodes", color="#cba6f7", fontsize=14, fontweight="bold")
    fig.tight_layout(pad=0.8); st.pyplot(fig, use_container_width=True)
    buf2=io.BytesIO(); fig.savefig(buf2,format="png",dpi=150,bbox_inches="tight",facecolor="#1e1e2e")
    plt.close(fig)
    st.download_button("💾 Télécharger", buf2.getvalue(), file_name="contours_all.png", mime="image/png")

# Tab 3
with tab_compare:
    c1,c2 = st.columns(2)
    m1 = c1.selectbox("Méthode A", method_names, index=0, key="m1")
    m2 = c2.selectbox("Méthode B", method_names, index=6, key="m2")
    d1,cm1=results[m1]; d2,cm2=results[m2]
    cp1 = cm1 if custom_cmap=="Auto" else custom_cmap
    cp2 = cm2 if custom_cmap=="Auto" else custom_cmap
    
    if wm_on_all and wm_text:
        import matplotlib.cm as cm
        # Pour d1
        if len(d1.shape) == 2:
            cmap_obj1 = cm.get_cmap(cp1)
            d1_rgb = (cmap_obj1(d1)[:,:,:3] * 255).astype(np.uint8)
        else:
            d1_rgb = to_uint8(d1)
        d1_show = apply_watermark(d1_rgb, wm_text, wm_color_rgb, wm_op, wm_size, wm_pos, wm_outline)
        
        # Pour d2
        if len(d2.shape) == 2:
            cmap_obj2 = cm.get_cmap(cp2)
            d2_rgb = (cmap_obj2(d2)[:,:,:3] * 255).astype(np.uint8)
        else:
            d2_rgb = to_uint8(d2)
        d2_show = apply_watermark(d2_rgb, wm_text, wm_color_rgb, wm_op, wm_size, wm_pos, wm_outline)
        
        is_compare_rgb = True
    else:
        d1_show = d1
        d2_show = d2
        is_compare_rgb = False

    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,6),facecolor="#1e1e2e")
    for ax,d,cm,name in [(ax1,d1_show,cp1,m1),(ax2,d2_show,cp2,m2)]:
        ax.set_facecolor("#1e1e2e")
        if is_compare_rgb:
            ax.imshow(d, interpolation="nearest")
        else:
            ax.imshow(d, cmap=cm, interpolation="nearest")
        ax.set_title(name,color="#cdd6f4",fontsize=12,fontweight="bold"); ax.axis("off")
    fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
    if d1.shape==d2.shape:
        diff=np.abs(d1.astype(np.float64)-d2.astype(np.float64))
        st.caption(f"**Diff. moyenne :** `{diff.mean():.4f}`  |  **Max :** `{diff.max():.4f}`")
        if st.checkbox("Carte de différence"):
            fd,ad = plt.subplots(figsize=(7,5),facecolor="#1e1e2e")
            ad.imshow(diff,cmap="plasma"); ad.set_title("|A−B|",color="#cdd6f4",fontsize=12)
            ad.axis("off"); plt.colorbar(ad.images[0],ax=ad,fraction=0.03)
            fd.tight_layout(); st.pyplot(fd,use_container_width=True); plt.close(fd)

# ── Tab 4 : Suppression Fond ──────────────────────────────────
with tab_bg:
    st.subheader(f"✂️ {bg_method}")

    # ── Ligne 1 : diagnostics intermédiaires
    st.markdown("#### 🔬 Étapes de traitement")
    d1, d2, d3, d4 = st.columns(4)

    with d1:
        st.caption("**Saillance spectrale**")
        fig_s, ax_s = plt.subplots(figsize=(4,3), facecolor="#1e1e2e")
        ax_s.imshow(sal_map, cmap="inferno"); ax_s.axis("off")
        fig_s.tight_layout(); st.pyplot(fig_s, use_container_width=True); plt.close(fig_s)

    with d2:
        st.caption("**Masque brut**")
        fig_r, ax_r = plt.subplots(figsize=(4,3), facecolor="#1e1e2e")
        ax_r.imshow(raw_mask, cmap="RdYlGn", vmin=0, vmax=1); ax_r.axis("off")
        fig_r.tight_layout(); st.pyplot(fig_r, use_container_width=True); plt.close(fig_r)

    with d3:
        st.caption("**Masque raffiné**")
        fig_m, ax_m = plt.subplots(figsize=(4,3), facecolor="#1e1e2e")
        ax_m.imshow(refined_mask, cmap="RdYlGn", vmin=0, vmax=1); ax_m.axis("off")
        fig_m.tight_layout(); st.pyplot(fig_m, use_container_width=True); plt.close(fig_m)

    with d4:
        st.caption("**Alpha matting**")
        fig_a, ax_a = plt.subplots(figsize=(4,3), facecolor="#1e1e2e")
        ax_a.imshow(alpha, cmap="gray", vmin=0, vmax=1); ax_a.axis("off")
        fig_a.tight_layout(); st.pyplot(fig_a, use_container_width=True); plt.close(fig_a)

    st.divider()

    # ── Ligne 2 : résultats finaux
    st.markdown("#### 🎨 Résultats")
    r1, r2, r3 = st.columns(3)

    result_colored  = apply_mask_soft(orig_rgb, alpha, bg_rgb_tuple)
    result_white    = apply_mask_soft(orig_rgb, alpha, (255,255,255))
    result_black    = apply_mask_soft(orig_rgb, alpha, (0,0,0))
    rgba_out        = to_rgba(orig_rgb, alpha)

    # ── Application des fonctions avancées ──
    
    # 1. Filtres artistiques
    if art_mode != "Aucun":
        result_colored = apply_artistic_mode(result_colored, art_mode)
        result_white   = apply_artistic_mode(result_white, art_mode)
        result_black   = apply_artistic_mode(result_black, art_mode)

    # 2. Reflet
    if do_reflect:
        result_colored = apply_reflection(result_colored, alpha, ref_op, ref_len)
        result_white   = apply_reflection(result_white, alpha, ref_op, ref_len)
        result_black   = apply_reflection(result_black, alpha, ref_op, ref_len)

    # 3. Ajustements (Luminosité, Saturation...)
    if adj_bright != 0 or adj_cont != 0 or adj_sat != 1.0 or adj_sharp != 0:
        result_colored = apply_image_adjustments(result_colored, adj_bright, adj_cont, adj_sat, adj_sharp)
        result_white   = apply_image_adjustments(result_white, adj_bright, adj_cont, adj_sat, adj_sharp)
        result_black   = apply_image_adjustments(result_black, adj_bright, adj_cont, adj_sat, adj_sharp)

    # 4. Recadrage automatique
    if do_crop:
        result_colored = auto_crop_to_subject(result_colored, alpha)
        result_white   = auto_crop_to_subject(result_white, alpha)
        result_black   = auto_crop_to_subject(result_black, alpha)

    # Application du filigrane (Watermark)
    if wm_text:
        result_colored = apply_watermark(result_colored, wm_text, wm_color_rgb, wm_op, wm_size, wm_pos, wm_outline)
        result_white   = apply_watermark(result_white,   wm_text, wm_color_rgb, wm_op, wm_size, wm_pos, wm_outline)
        result_black   = apply_watermark(result_black,   wm_text, wm_color_rgb, wm_op, wm_size, wm_pos, wm_outline)
        rgb_wm = apply_watermark(rgba_out[:,:,:3], wm_text, wm_color_rgb, wm_op, wm_size, wm_pos, wm_outline)
        rgba_out = np.dstack([rgb_wm, rgba_out[:,:,3]])

    with r1:
        st.caption("**Original**")
        st.image(orig_rgb, use_container_width=True)
    with r2:
        st.caption(f"**Fond : {bg_color_hex}**")
        st.image(result_colored, use_container_width=True)
    with r3:
        st.caption("**Fond blanc**")
        st.image(result_white, use_container_width=True)

    st.divider()

    # Statistiques
    fg_px = int(refined_mask.sum())
    bg_px = refined_mask.size - fg_px
    cov   = fg_px / refined_mask.size * 100
    qual  = float(np.mean(alpha[refined_mask==1]))

    sc1,sc2,sc3,sc4 = st.columns(4)
    sc1.metric("Pixels sujet",  f"{fg_px:,}")
    sc2.metric("Pixels fond",   f"{bg_px:,}")
    sc3.metric("Couverture",    f"{cov:.1f} %")
    sc4.metric("Qualité alpha", f"{qual:.2f}")

    st.divider()

    # Exports
    e1, e2, e3 = st.columns(3)
    buf_rgba = io.BytesIO(); Image.fromarray(rgba_out).save(buf_rgba, format="PNG")
    buf_col  = io.BytesIO(); Image.fromarray(result_colored).save(buf_col, format="PNG")
    buf_wh   = io.BytesIO(); Image.fromarray(result_white).save(buf_wh, format="PNG")

    e1.download_button("💾 PNG transparent",   buf_rgba.getvalue(), "sujet_transparent.png", "image/png", use_container_width=True)
    e2.download_button("💾 Fond coloré",       buf_col.getvalue(),  "sujet_fond_couleur.png","image/png", use_container_width=True)
    e3.download_button("💾 Fond blanc",        buf_wh.getvalue(),   "sujet_fond_blanc.png",  "image/png", use_container_width=True)

    if not REMBG_AVAILABLE:
        st.info("💡 Pour la meilleure qualité : `pip install rembg onnxruntime` — réseau U²-Net (deep learning)", icon="🤖")

# ── Tab 5 : Poisson Blending ──────────────────────────────────
with tab_poisson:
    st.subheader("🐟 Incrustation Seamless (Poisson Blending)")
    if 'target_uploaded' not in locals() or target_uploaded is None:
        st.info("Veuillez charger une 'Image Cible (Fond)' dans la barre latérale pour utiliser l'incrustation.")
    else:
        target_bytes = np.frombuffer(target_uploaded.read(), np.uint8)
        target_color = cv2.imdecode(target_bytes, cv2.IMREAD_COLOR)
        if target_color is not None:
            target_rgb = cv2.cvtColor(target_color, cv2.COLOR_BGR2RGB)
            
            c1, c2 = st.columns(2)
            p_mode = c1.selectbox("Mode de fusion", ["Copier-Coller Simple (Alpha)", "Normal", "Gradient Max", "Gradient Min"])
            
            st.markdown("### 🛠️ Transformation & Couleur")
            tc1, tc2, tc3 = st.columns(3)
            p_scale = tc1.slider("Échelle (Taille)", 0.05, 5.0, 1.0, 0.05)
            p_angle = tc2.slider("Rotation", -180, 180, 0, 5)
            do_color_match = tc3.checkbox("Harmoniser lumières", value=False)

            if "Copier-Coller Simple" in p_mode:
                opacity = c1.slider("Opacité de l'objet", 0.0, 1.0, 1.0, 0.05)
            else:
                opacity = 1.0

            st.markdown("### 🌑 Ombre Portée")
            sh1, sh2, sh3, sh4 = st.columns(4)
            do_shadow = sh1.checkbox("Activer l'ombre", value=False)
            sh_blur = sh2.slider("Flou ombre", 0, 50, 15) if do_shadow else 15
            sh_op   = sh3.slider("Opacité ombre", 0.0, 1.0, 0.4, 0.05) if do_shadow else 0.4
            sh_off  = sh4.slider("Décalage XY", -50, 50, 10) if do_shadow else 10
            
            y_idx, x_idx = np.where(refined_mask == 1)
            if len(y_idx) == 0:
                st.error("Le masque du premier plan est vide.")
            else:
                y_min, y_max = y_idx.min(), y_idx.max()
                x_min, x_max = x_idx.min(), x_idx.max()
                
                # Patch original serré
                src_patch_raw = orig_rgb[y_min:y_max+1, x_min:x_max+1]
                alpha_patch_raw = alpha[y_min:y_max+1, x_min:x_max+1]
                
                # 1. Transformation (Échelle / Rotation)
                src_patch, alpha_patch = transform_patch(src_patch_raw, alpha_patch_raw, p_scale, p_angle)
                
                # --- AUTO-FIT : Redimensionnement automatique si trop grand pour le fond ---
                bh, bw = target_rgb.shape[:2]
                ph, pw = src_patch.shape[:2]
                if ph > bh or pw > bw:
                    fit_scale = min(bh / ph, bw / pw) * 0.98
                    src_patch, alpha_patch = transform_patch(src_patch, alpha_patch, fit_scale, 0)
                    st.warning(f"⚠️ L'objet était trop grand ({pw}x{ph}) pour le fond ({bw}x{bh}). Il a été réduit à {src_patch.shape[1]}x{src_patch.shape[0]}.")
                # -------------------------------------------------------------------------

                # 2. Harmonisation des couleurs (si activée)
                if do_color_match:
                    src_patch = color_transfer(src_patch, target_rgb)
                
                m_p, n_p = src_patch.shape[:2]
                msk_patch = (alpha_patch > 0.5).astype(np.uint8)
                msk_patch[0,:] = 0; msk_patch[-1,:] = 0; msk_patch[:,0] = 0; msk_patch[:,-1] = 0
                
                max_x = target_rgb.shape[1] - n_p
                max_y = target_rgb.shape[0] - m_p
                
                if max_x > 0:
                    pos_x = c2.slider("Position X", 0, max_x, 0)
                else:
                    pos_x = 0
                        
                if max_y > 0:
                    pos_y = c2.slider("Position Y", 0, max_y, 0)
                else:
                    pos_y = 0
                
                tgt_patch = target_rgb[pos_y:pos_y+m_p, pos_x:pos_x+n_p]
                    
                # --- APERÇU EN DIRECT (AVANT FUSION) ---
                st.markdown("#### 👁️ Aperçu du positionnement")
                preview_bg = target_rgb.copy().astype(np.float32)
                a3 = np.dstack([alpha_patch]*3) * opacity
                
                # Layer 1: Shadow (si activée)
                if do_shadow:
                    shadow_mask = generate_shadow(alpha_patch, sh_blur, sh_op, sh_off, sh_off)
                    s3 = np.dstack([shadow_mask]*3)
                    # On applique l'ombre sur le fond
                    preview_bg[pos_y:pos_y+m_p, pos_x:pos_x+n_p] = \
                        preview_bg[pos_y:pos_y+m_p, pos_x:pos_x+n_p] * (1 - s3) # Assombrissement
                
                # Layer 2: Object
                preview_patch = src_patch.astype(np.float32) * a3 + preview_bg[pos_y:pos_y+m_p, pos_x:pos_x+n_p] * (1 - a3)
                preview_bg[pos_y:pos_y+m_p, pos_x:pos_x+n_p] = preview_patch
                
                st.image(np.clip(preview_bg, 0, 255).astype(np.uint8), use_container_width=True, caption="Position de l'objet avant fusion")
                
                if st.button("🚀 Lancer l'Incrustation Poisson") and tgt_patch is not None:
                    if m_p * n_p > 800000:
                        st.warning("⚠️ La zone à fusionner est très grande et peut prendre du temps ou saturer la mémoire.")
                    with st.spinner("Résolution de l'équation de Poisson..."):
                        try:
                            # a3 et clean_src_patch utilisent les versions TRANSFORMÉES
                            a3 = np.dstack([alpha_patch]*3) * opacity
                            
                            # On repart d'un fond propre pour la fusion finale
                            bg_for_fusion = tgt_patch.copy().astype(np.float32)
                            
                            if do_shadow:
                                shadow_mask = generate_shadow(alpha_patch, sh_blur, sh_op, sh_off, sh_off)
                                s3 = np.dstack([shadow_mask]*3)
                                bg_for_fusion = bg_for_fusion * (1 - s3)

                            clean_src_patch = src_patch.astype(np.float32) * a3 + bg_for_fusion * (1 - a3)
                            clean_src_patch = np.clip(clean_src_patch, 0, 255).astype(np.uint8)

                            if "Copier-Coller Simple" in p_mode:
                                result_patch = clean_src_patch
                            else:
                                p_mode_clean = p_mode.split(" ")[0] + (" Max" if "Max" in p_mode else "") + (" Min" if "Min" in p_mode else "")
                                result_patch = poisson_blend_patch(clean_src_patch, bg_for_fusion.astype(np.uint8), msk_patch, p_mode_clean.strip())
                                
                            final_bg = target_rgb.copy()
                            final_bg[pos_y:pos_y+m_p, pos_x:pos_x+n_p] = result_patch
                            
                            # Filigrane sur la composition finale
                            if wm_text:
                                final_bg = apply_watermark(final_bg, wm_text, wm_color_rgb, wm_op, wm_size, wm_pos, wm_outline)
                            
                            st.image(final_bg, use_container_width=True)
                            
                            buf_poisson = io.BytesIO()
                            Image.fromarray(final_bg).save(buf_poisson, format="PNG")
                            st.download_button("💾 Télécharger Résultat", buf_poisson.getvalue(), "poisson_blending.png", "image/png")
                        except Exception as e:
                            st.error(f"Erreur lors de la fusion : {e}")

# ── Tab 6 : Avant/Après (Wipe) ──────────────────────────────────
with tab_wipe:
    st.subheader("🌓 Comparaison Avant / Après")
    st.markdown("Comparez l'image originale avec le résultat détouré.")

    # 1. Sélection du style de fond pour le résultat
    bg_preview_style = st.radio("Style de fond pour l'aperçu", 
                                ["Damier transparent (Simulé)", "Couleur de fond sélectionnée", "Fond Blanc", "Fond Noir", "Transparent (Sans fond)"],
                                horizontal=True, key="wipe_bg_style")
    
    # 2. Préparation du résultat détouré selon le style
    result_colored_wipe = apply_mask_soft(orig_rgb, alpha, bg_rgb_tuple)
    
    # Réappliquer les filtres artistiques, ajustements, etc.
    if art_mode != "Aucun":
        result_colored_wipe = apply_artistic_mode(result_colored_wipe, art_mode)
    if do_reflect:
        result_colored_wipe = apply_reflection(result_colored_wipe, alpha, ref_op, ref_len)
    if adj_bright != 0 or adj_cont != 0 or adj_sat != 1.0 or adj_sharp != 0:
        result_colored_wipe = apply_image_adjustments(result_colored_wipe, adj_bright, adj_cont, adj_sat, adj_sharp)
    if wm_text:
        result_colored_wipe = apply_watermark(result_colored_wipe, wm_text, wm_color_rgb, wm_op, wm_size, wm_pos, wm_outline)

    # Déterminer l'image "Après"
    if bg_preview_style == "Couleur de fond sélectionnée":
        after_img = result_colored_wipe
    elif bg_preview_style == "Fond Blanc":
        after_img = apply_mask_soft(orig_rgb, alpha, (255,255,255))
        if art_mode != "Aucun": after_img = apply_artistic_mode(after_img, art_mode)
        if do_reflect: after_img = apply_reflection(after_img, alpha, ref_op, ref_len)
        if adj_bright != 0 or adj_cont != 0 or adj_sat != 1.0 or adj_sharp != 0:
            after_img = apply_image_adjustments(after_img, adj_bright, adj_cont, adj_sat, adj_sharp)
        if wm_text: after_img = apply_watermark(after_img, wm_text, wm_color_rgb, wm_op, wm_size, wm_pos, wm_outline)
    elif bg_preview_style == "Fond Noir":
        after_img = apply_mask_soft(orig_rgb, alpha, (0,0,0))
        if art_mode != "Aucun": after_img = apply_artistic_mode(after_img, art_mode)
        if do_reflect: after_img = apply_reflection(after_img, alpha, ref_op, ref_len)
        if adj_bright != 0 or adj_cont != 0 or adj_sat != 1.0 or adj_sharp != 0:
            after_img = apply_image_adjustments(after_img, adj_bright, adj_cont, adj_sat, adj_sharp)
        if wm_text: after_img = apply_watermark(after_img, wm_text, wm_color_rgb, wm_op, wm_size, wm_pos, wm_outline)
    elif bg_preview_style == "Transparent (Sans fond)":
        after_img = to_rgba(orig_rgb, alpha)
        if wm_text:
            rgb_wm_rgba = apply_watermark(after_img[:,:,:3], wm_text, wm_color_rgb, wm_op, wm_size, wm_pos, wm_outline)
            after_img = np.dstack([rgb_wm_rgba, after_img[:,:,3]])
    else: # Damier transparent (Simulé)
        h_im, w_im = orig_rgb.shape[:2]
        checker = np.zeros((h_im, w_im, 3), dtype=np.uint8)
        grid_size = 20
        for r in range(0, h_im, grid_size):
            for c_idx in range(0, w_im, grid_size):
                if ((r // grid_size) + (c_idx // grid_size)) % 2 == 0:
                    checker[r:r+grid_size, c_idx:c_idx+grid_size] = 240
                else:
                    checker[r:r+grid_size, c_idx:c_idx+grid_size] = 200
        a3 = alpha[:,:,np.newaxis]
        after_img = np.clip(orig_rgb.astype(np.float32) * a3 + checker.astype(np.float32) * (1 - a3), 0, 255).astype(np.uint8)
        if art_mode != "Aucun": after_img = apply_artistic_mode(after_img, art_mode)
        if do_reflect: after_img = apply_reflection(after_img, alpha, ref_op, ref_len)
        if adj_bright != 0 or adj_cont != 0 or adj_sat != 1.0 or adj_sharp != 0:
            after_img = apply_image_adjustments(after_img, adj_bright, adj_cont, adj_sat, adj_sharp)
        if wm_text: after_img = apply_watermark(after_img, wm_text, wm_color_rgb, wm_op, wm_size, wm_pos, wm_outline)

    # 3. Choix de la présentation
    presentation_mode = st.radio("Type de visualisation", ["Curseur de séparation glissant (Interactif)", "Superposition progressive", "Côte à côte"], horizontal=True)

    if presentation_mode == "Côte à côte":
        col_bef, col_aft = st.columns(2)
        with col_bef:
            st.caption("👈 Avant (Original)")
            st.image(orig_rgb, use_container_width=True)
        with col_aft:
            st.caption("👉 Après (Détouré)")
            st.image(after_img, use_container_width=True)

    elif presentation_mode == "Superposition progressive":
        blend_val = st.slider("Faites glisser pour ajuster le fondu (Original ➡️ Traité)", 0.0, 1.0, 0.5, 0.05)
        # Ensure we have RGB for blending
        after_rgb = after_img[:,:,:3] if len(after_img.shape) == 3 and after_img.shape[2] == 4 else after_img
        blended = cv2.addWeighted(orig_rgb, 1.0 - blend_val, after_rgb, blend_val, 0)
        st.image(blended, use_container_width=True, caption=f"Original ({int((1-blend_val)*100)}%) / Traité ({int(blend_val*100)}%)")

    else: # Curseur de séparation glissant (Interactif)
        # Convert both images to base64
        import base64
        
        # Helper function to resize for fast loading/rendering inside HTML
        def get_b64_img(image):
            h_orig, w_orig = image.shape[:2]
            max_dim = 800
            if max(h_orig, w_orig) > max_dim:
                scale = max_dim / max(h_orig, w_orig)
                resized = cv2.resize(image, (int(w_orig * scale), int(h_orig * scale)), interpolation=cv2.INTER_AREA)
            else:
                resized = image
            pil_i = Image.fromarray(resized)
            buf = io.BytesIO()
            pil_i.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        
        # Prepare RGB versions
        after_rgb_for_slider = after_img[:,:,:3] if len(after_img.shape) == 3 and after_img.shape[2] == 4 else after_img
        
        try:
            b64_before = get_b64_img(orig_rgb)
            b64_after = get_b64_img(after_rgb_for_slider)
            
            # HTML / CSS / JS Split Slider
            slider_html = f"""
            <div class="slider-container" id="slider-container" style="position: relative; width: 100%; max-width: 800px; margin: auto; overflow: hidden; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); user-select: none;">
              <img src="data:image/png;base64,{b64_before}" style="display: block; width: 100%; height: auto; pointer-events: none;" />
              <div class="overlay" id="overlay" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; overflow: hidden; clip-path: polygon(50% 0, 100% 0, 100% 100%, 50% 100%);">
                <img src="data:image/png;base64,{b64_after}" style="display: block; width: 100%; height: auto; pointer-events: none;" />
              </div>
              <div class="slider-bar" id="slider-bar" style="position: absolute; top: 0; bottom: 0; left: 50%; width: 4px; background: #cba6f7; cursor: ew-resize; z-index: 10;">
                <div class="slider-handle" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 42px; height: 42px; background: #cba6f7; color: #1e1e2e; border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 10px rgba(0,0,0,0.4); font-weight: bold; font-size: 20px;">🌓</div>
              </div>
            </div>
            
            <script>
            const container = document.getElementById('slider-container');
            const overlay = document.getElementById('overlay');
            const slider = document.getElementById('slider-bar');
            let isSliding = false;
            
            function move(clientX) {{
              const rect = container.getBoundingClientRect();
              let pos = (clientX - rect.left) / rect.width;
              if (pos < 0) pos = 0;
              if (pos > 1) pos = 1;
              overlay.style.clipPath = `polygon(${{pos*100}}% 0, 100% 0, 100% 100%, ${{pos*100}}% 100%)`;
              slider.style.left = `${{pos * 100}}%`;
            }}
            
            container.addEventListener('mousedown', (e) => {{
              isSliding = true;
              move(e.clientX);
            }});
            window.addEventListener('mouseup', () => isSliding = false);
            window.addEventListener('mousemove', (e) => {{
              if (!isSliding) return;
              move(e.clientX);
            }});
            
            container.addEventListener('touchstart', (e) => {{
              isSliding = true;
              move(e.touches[0].clientX);
            }});
            window.addEventListener('touchend', () => isSliding = false);
            window.addEventListener('touchmove', (e) => {{
              if (!isSliding) return;
              move(e.touches[0].clientX);
            }});
            </script>
            """
            st.components.v1.html(slider_html, height=550)
            st.caption("<p align='center'>Faites glisser le curseur 🌓 au milieu de l'image de gauche à droite</p>", unsafe_allow_html=True)
        except Exception as slider_err:
            st.error(f"Impossible de charger le curseur interactif : {slider_err}")
            # Fallback
            st.image(orig_rgb, caption="Original", use_container_width=True)
            st.image(after_img, caption="Détouré", use_container_width=True)