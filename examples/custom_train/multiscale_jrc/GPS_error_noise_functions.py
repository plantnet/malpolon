import numpy as np
import pandas as pd
from pyproj import Transformer


# WGS84 → ETRS89 / LAEA Europe (EPSG:3035)
to_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
to_4326 = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)


def add_stale_locations(df, p=0.01, seed=None):
    """Swap df lines to simulate old locations not being updated by sensors."""
    rng = np.random.default_rng(seed)
    df = df.copy()

    mask = rng.random(len(df)) < p
    idx = np.where(mask)[0]

    if len(idx) == 0:
        return df

    # projection en 3035
    x, y = to_3035.transform(df["lon"].values, df["lat"].values)

    # tirage aléatoire
    replacement_idx = rng.integers(0, len(df), size=len(idx))

    x_new = x.copy()
    y_new = y.copy()

    x_new[idx] = x[replacement_idx]
    y_new[idx] = y[replacement_idx]

    # back transform
    lon_new, lat_new = to_4326.transform(x_new, y_new)

    df["lon_noisy"] = lon_new
    df["lat_noisy"] = lat_new

    return df


def add_heavy_tail_noise(df, p=0.01, median_m=100000, sigma=1.5, seed=None):
    rng = np.random.default_rng(seed)
    df = df.copy()

    mask = rng.random(len(df)) < p
    idx = np.where(mask)[0]

    if len(idx) == 0:
        return df

    # projection
    x, y = to_3035.transform(df["lon"].values, df["lat"].values)

    # distances (log-normal)
    distances = rng.lognormal(mean=np.log(median_m), sigma=sigma, size=len(idx))
    angles = rng.uniform(0, 2*np.pi, size=len(idx))

    dx = distances * np.cos(angles)
    dy = distances * np.sin(angles)
    # Clipping with roughly min/max x and y extent (taken from Qgis by zooming out of a map in CRS EPSG:3035)
    dx = np.clip(dx, -16578866, 17786310)
    dy = np.clip(dy, -3449850, 17093779)

    x_new = x.copy()
    y_new = y.copy()

    x_new[idx] += dx
    y_new[idx] += dy

    # back transform
    lon_new, lat_new = to_4326.transform(x_new, y_new)

    df["lon_noisy"] = lon_new
    df["lat_noisy"] = lat_new

    return df


def add_swap_latlon(df, p=0.005, seed=None):
    rng = np.random.default_rng(seed)
    df = df.copy()

    mask = rng.random(len(df)) < p

    lat = df.loc[mask, "lat"].values
    lon = df.loc[mask, "lon"].values

    df.loc[mask, "lat_noisy"] = lon
    df.loc[mask, "lon_noisy"] = lat

    return df


def add_sign_flip(df, p=0.005, seed=None):
    rng = np.random.default_rng(seed)
    df = df.copy()

    mask = rng.random(len(df)) < p
    n = mask.sum()

    if n == 0:
        return df

    flip_lat = rng.random(n) < 0.5

    lat = df.loc[mask, "lat"].values.copy()
    lon = df.loc[mask, "lon"].values.copy()

    lat[flip_lat] *= -1
    lon[~flip_lat] *= -1

    df.loc[mask, "lat_noisy"] = lat
    df.loc[mask, "lon_noisy"] = lon

    return df


def add_rounding(df, p=0.01, decimals=2, seed=None):
    rng = np.random.default_rng(seed)
    df = df.copy()

    mask = rng.random(len(df)) < p

    df.loc[mask, "lat_noisy"] = df.loc[mask, "lat"].round(decimals)
    df.loc[mask, "lon_noisy"] = df.loc[mask, "lon"].round(decimals)

    return df

def add_rounding_variable_decimals(df, p=0.01, decimal_sup=2, decimal_inf=5, seed=None):
    rng = np.random.default_rng(seed)
    df = df.copy()

    mask = rng.random(len(df)) < p
    idx = df.index[mask]# np.where(mask)[0]

    if len(idx) == 0:
        return df

    lat = df.loc[idx, "lat"].values
    lon = df.loc[idx, "lon"].values

    # nombre de décimales dispo (approx via string)
    def count_decimals(x):
        s = f"{x:.10f}".rstrip("0").rstrip(".")
        if "." in s:
            return len(s.split(".")[1])
        return 0

    lat_dec = np.array([count_decimals(x) for x in lat])
    lon_dec = np.array([count_decimals(x) for x in lon])

    # tirer aléatoirement le nombre de décimales cible
    target_decimals = rng.integers(decimal_sup, decimal_inf, size=len(idx))

    lat_new = lat.copy()
    lon_new = lon.copy()

    for i in range(len(idx)):
        # on ne round que si ça enlève effectivement de l'info
        if lat_dec[i] > target_decimals[i]:
            lat_new[i] = np.round(lat[i], target_decimals[i])
        if lon_dec[i] > target_decimals[i]:
            lon_new[i] = np.round(lon[i], target_decimals[i])

    df.loc[idx, "lat_noisy"] = lat_new
    df.loc[idx, "lon_noisy"] = lon_new

    return df

def generate_radial_points_concentrated_inside_ring(
    center=(0.0, 0.0),
    n=1000,
    min_radius=75,
    max_radius=125,
    inner_scale=20,
    outer_scale=30,
    seed=None
):
    """
    Generate 2D points with:
    - 80% in a ring [min_radius, max_radius]
    - 10% inside with decreasing density toward center
    - 10% outside with decreasing density outward

    Returns a DataFrame with columns ['x', 'y']
    """

    rng = np.random.default_rng(seed)

    # counts
    n_ring = int(0.8 * n)
    n_inner = int(0.1 * n)
    n_outer = n - n_ring - n_inner

    # -----------------------
    # 1. RING (uniform area)
    # -----------------------
    u = rng.uniform(0, 1, n_ring)
    r_ring = np.sqrt(
        u * (max_radius**2 - min_radius**2) + min_radius**2
    )

    theta_ring = rng.uniform(0, 2*np.pi, n_ring)

    # -----------------------
    # 2. INNER (decreasing to center)
    # -----------------------
    r_inner = min_radius - rng.exponential(scale=inner_scale, size=n_inner)
    r_inner = np.clip(r_inner, 0, min_radius)

    theta_inner = rng.uniform(0, 2*np.pi, n_inner)

    # -----------------------
    # 3. OUTER (decreasing outward)
    # -----------------------
    r_outer = max_radius + rng.exponential(scale=outer_scale, size=n_outer)
    theta_outer = rng.uniform(0, 2*np.pi, n_outer)

    # -----------------------
    # Combine
    # -----------------------
    r = np.concatenate([r_ring, r_inner, r_outer])
    theta = np.concatenate([theta_ring, theta_inner, theta_outer])

    # Convert to Cartesian
    cx, cy = center
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)

    df = pd.DataFrame({"lon_noise": x, "lat_noise": y})

    return df

def generate_radial_points_concentrated_beyond_ring(
    center=(0.0, 0.0),
    n=1000,
    min_radius=75,
    inner_scale=20,
    outer_scale=50,
    seed=None
):
    """
    Generate 2D points with:
    - 90% outside min_radius with decreasing probability outward
    - 10% inside min_radius with decreasing probability toward center

    Returns a DataFrame with columns ['x', 'y']
    """

    rng = np.random.default_rng(seed)

    # counts
    n_outer = int(0.9 * n)
    n_inner = n - n_outer

    # -----------------------
    # 1. OUTER (90%)
    # decreasing outward
    # -----------------------
    r_outer = min_radius + rng.exponential(scale=outer_scale, size=n_outer)
    theta_outer = rng.uniform(0, 2*np.pi, n_outer)

    # -----------------------
    # 2. INNER (10%)
    # decreasing toward center
    # -----------------------
    r_inner = min_radius - rng.exponential(scale=inner_scale, size=n_inner)
    r_inner = np.clip(r_inner, 0, min_radius)
    theta_inner = rng.uniform(0, 2*np.pi, n_inner)

    # -----------------------
    # Combine
    # -----------------------
    r = np.concatenate([r_outer, r_inner])
    theta = np.concatenate([theta_outer, theta_inner])

    # Convert to Cartesian
    cx, cy = center
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)

    df = pd.DataFrame({"x": x, "y": y})

    return df

def add_ring_exponential_noise(df, noise_meters=100000, min_radius=75, max_radius=125, inner_scale=20, outer_scale=50, seed=None, noise_version='beyond_ring'):
    """
    Generate noisy GPS points around a given (lon, lat) in meters using EPSG:3035 projection.

    Parameters
    ----------
    lon, lat : float
        Original point coordinates in WGS84.
    n : int
        Number of noisy points to generate.
    noise_meters : float
        Standard deviation of noise in meters applied to x and y.
    seed : int or None
        Random seed.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns ["lon_noisy", "lat_noisy"] containing noisy points.
    """
    rng = np.random.default_rng(seed)

    # Convert to projected coordinates
    x, y = to_3035.transform(df['lon'].values, df['lat'].values)

    # Generate independent noise in meters
    if noise_version == 'inside_ring':
        df_noise = generate_radial_points_concentrated_inside_ring(center=(0.0, 0.0), n=len(df), min_radius=min_radius, max_radius=max_radius, inner_scale=inner_scale, outer_scale=30, seed=None)*noise_meters/100
    elif noise_version == 'beyond_ring':
        df_noise = generate_radial_points_concentrated_beyond_ring(center=(0.0, 0.0), n=len(df), min_radius=min_radius, inner_scale=inner_scale, outer_scale=outer_scale, seed=None)*noise_meters/100

    # Apply noise
    x_noisy = x + df_noise['x']
    y_noisy = y + df_noise['y']

    # Convert back to lon/lat
    lon_noisy, lat_noisy = to_4326.transform(x_noisy, y_noisy)

    # Store in DataFrame
    df = pd.DataFrame({
        "lon_noisy": lon_noisy,
        "lat_noisy": lat_noisy
    })

    return df