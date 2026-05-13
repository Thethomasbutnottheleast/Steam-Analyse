import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

def calculate_mca_adjustment(lambdas, q):
    threshold = 1 / q
    adjusted_lambdas = []
    
    for l in lambdas:
        if l > threshold:
            # Applying Benzecri's Formula
            adj = ((q / (q - 1))**2) * ((l - threshold)**2)
            adjusted_lambdas.append(adj)
        else:
            # Dimensions below 1/Q are considered noise and ignored in adjustment
            adjusted_lambdas.append(0)
            
    # Calculate percentages based on the sum of adjusted inertias
    total_adj_inertia = sum(adjusted_lambdas)
    adj_percentages = [(a / total_adj_inertia) * 100 for a in adjusted_lambdas]
    
    return adjusted_lambdas, adj_percentages

def eta2_binary_one_axis(coord_axis, binary_var):
    data = pd.DataFrame({
        "coord": coord_axis,
        "g": binary_var
    }).dropna()

    y = data["coord"].to_numpy()
    g = data["g"].to_numpy()

    grand_mean = y.mean()
    ss_total = np.sum((y - grand_mean) ** 2)

    if ss_total == 0:
        return np.nan

    ss_between = 0.0
    for level in np.unique(g):
        yk = y[g == level]
        nk = len(yk)
        mk = yk.mean()
        ss_between += nk * (mk - grand_mean) ** 2

    return ss_between / ss_total
    
def eta2_genres_table(coords, genre_dummy):
    """
    Compute eta^2 between each MCA axis and each genre dummy variable.
    """

    result = pd.DataFrame(
        index=coords.columns,
        columns=genre_dummy.columns,
        dtype=float
    )

    for axis in coords.columns:
        for genre in genre_dummy.columns:
            result.loc[axis, genre] = eta2_binary_one_axis(
                coords[axis],
                genre_dummy[genre]
            )

    return result

def plot_eta2_variables(eta2_df, dim_x=0, dim_y=1, title="Variables representation"):
    """
    Plot genres using eta² on two MCA dimensions.
    
    eta2_df: DataFrame
        rows = dimensions, columns = variables/genres
    dim_x, dim_y: int
        dimensions to plot (row indices)
    """

    x = eta2_df.loc[dim_x]
    y = eta2_df.loc[dim_y]

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(x, y)

    for var in eta2_df.columns:
        ax.text(x[var], y[var], var, fontsize=11)

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.axvline(0, linestyle="--", linewidth=1)

    ax.set_xlabel(f"Dim {dim_x}")
    ax.set_ylabel(f"Dim {dim_y}")
    ax.set_title(title)

    # optional: keep only positive quadrant if you want slide-like appearance
    ax.set_xlim(left=min(0, x.min() - 0.02))
    ax.set_ylim(bottom=min(0, y.min() - 0.02))

    plt.show()
    

def plot_eta2_3d(eta2_df, dim_x=0, dim_y=1, dim_z=2,write=False, name=None):
    # Build plotting dataframe
    df_plot = pd.DataFrame({
        "Genre": eta2_df.columns,
        f"Dim {dim_x}": eta2_df.loc[dim_x].values,
        f"Dim {dim_y}": eta2_df.loc[dim_y].values,
        f"Dim {dim_z}": eta2_df.loc[dim_z].values,
    })

    fig = px.scatter_3d(
        df_plot,
        x=f"Dim {dim_x}",
        y=f"Dim {dim_y}",
        z=f"Dim {dim_z}",
        text="Genre",
        hover_name="Genre"
    )

    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        title="3D representation of squared correlation ratios (eta²)",
        scene=dict(
            xaxis_title=f"Dim {dim_x}",
            yaxis_title=f"Dim {dim_y}",
            zaxis_title=f"Dim {dim_z}",
        ),
        width=900,
        height=700
    )

    fig.show()
    if write:
        fig.write_html(
            f"{name}.html",
            include_plotlyjs="cdn",
            full_html=True
        )

def show_contrib_cos2_mod(mca, data, top, top_n):
    """
    Show the top contribution and cos2 of the modalities
    top : number of dimensions to show
    top_n : number of top modalities with highest cos2 to show
    """

    contrib = mca.column_contributions_
    # --- matrices ---
    cos2 = mca.column_cosine_similarities(data)
    mat_cos2 = cos2.iloc[:, :(top+1)]
    mat_contrib = contrib.iloc[:, :(top+1)]
    
    # --- choose rows to display: top from either contrib or cos2 ---
    top_rows = set()
    
    for c in mat_cos2.columns:
        top_rows.update(mat_contrib[c].nlargest(top_n).index)
        top_rows.update(mat_cos2[mat_cos2.index.str.endswith("__1")][c].nlargest(top_n).index)
    
    top_rows = list(top_rows)
    
    # optional: order rows by strongest value overall
    row_order = (
        pd.concat(
            [mat_cos2.loc[top_rows].max(axis=1), mat_contrib.loc[top_rows].max(axis=1)],
            axis=1
        )
        .max(axis=1)
        .sort_values(ascending=False)
        .index
    )
    
    mat_cos2_plot = mat_cos2.loc[row_order]
    mat_contrib_plot = mat_contrib.loc[row_order]
    
    # --- plot side by side ---
    fig, axes = plt.subplots(
        1, 2, figsize=(16, max(6, 0.4 * len(row_order))), sharey=True
    )
    
    sns.heatmap(
        mat_contrib_plot,
        annot=True, fmt=".2f", cmap="magma",
        ax=axes[0], cbar=True
    )
    axes[0].set_title("Contribution")
    axes[0].set_xlabel("Dimensions")
    axes[0].set_ylabel("Genres")
    
    sns.heatmap(
        mat_cos2_plot,
        annot=True, fmt=".2f", cmap="Blues",
        ax=axes[1], cbar=True
    )
    axes[1].set_title("Cos²")
    axes[1].set_xlabel("Dimensions")
    axes[1].set_ylabel("")
    
    plt.tight_layout()
    plt.show()

def variable_contributions_from_modalities(mod_contrib, sep="_"):
    """
    Compute contribution per variable from MCA modality contributions.

    Parameters
    ----------
    mod_contrib : pd.DataFrame
        MCA column/modalities contributions.
        Rows = modalities, e.g. Action_0, Action_1.
        Columns = dimensions.
    
    sep : str
        Separator between variable name and modality level.
        Default is "_".

    Returns
    -------
    var_contrib : pd.DataFrame
        Rows = variables, e.g. Action, Adventure.
        Columns = dimensions.
        Values = summed contribution of all modalities of each variable.
    """

    mod_contrib = mod_contrib.copy()

    # Extract variable name by removing the last part after "_"
    variables = mod_contrib.index.to_series().astype(str).apply(
        lambda x: sep.join(x.split(sep)[:-1])
    )

    var_contrib = mod_contrib.groupby(variables).sum()

    return var_contrib

def select_modalities_by_axis(coords, cos2,dims=(0, 1),top_n=5):
    '''
    Choisir les modalités bien représentées
    '''
    selected = set()
    for d in dims:
        selected.update(cos2[d].nlargest(top_n).index)
    return coords.loc[sorted(selected)]
def plot_mca_coords(ind_coords=None, active_coords=None, supp_coords=None,
                    dim_x=0, dim_y=1, figsize=(10, 8), title=None,
                    ind_alpha=0.35, ind_size=12):
    fig, ax = plt.subplots(figsize=figsize)

    # Individuals in background
    if ind_coords is not None:
        ax.scatter(
            ind_coords[dim_x], ind_coords[dim_y],
            s=ind_size, color="grey", alpha=ind_alpha,
            label="Individuals", zorder=1
        )

    # Active modalities
    if active_coords is not None:
        ax.scatter(
            active_coords[dim_x], active_coords[dim_y],
            s=50, label="Active modalities", zorder=2
        )

        for label, (x, y) in active_coords[[dim_x, dim_y]].iterrows():
            ax.text(x, y, label, fontsize=9, alpha=0.9, zorder=3, color='blue')

    # Supplementary modalities
    if supp_coords is not None:
        ax.scatter(
            supp_coords[dim_x], supp_coords[dim_y],
            s=90, marker='D', label="Supplementary modalities", zorder=4
        )

        for label, (x, y) in supp_coords[[dim_x, dim_y]].iterrows():
            ax.text(x, y, label, fontsize=10, zorder=5, color='orange')

    # Axes
    ax.axhline(0, color='gray', linewidth=1)
    ax.axvline(0, color='gray', linewidth=1)

    ax.set_xlabel(f"Dimension {dim_x}")
    ax.set_ylabel(f"Dimension {dim_y}")
    ax.set_title(title if title else f"MCA coordinates: Dim {dim_x} vs Dim {dim_y}")
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot3D_ind_var_mca(mca, data, dim0=0, dim1=1, dim2=2, write = False, name=None):
    
    # --- Individual coordinates ---
    df_ind = mca.row_coordinates(data).copy()
    df_ind = df_ind.iloc[:, [dim0, dim1, dim2]]
    df_ind.columns = [f"Dim{dim0}", f"Dim{dim1}", f"Dim{dim2}"]
    
    # --- Modality coordinates ---
    df_var = mca.column_coordinates(data).copy()
    df_var = df_var.iloc[:, :3]
    df_var.columns = [f"Dim{dim0}", f"Dim{dim1}", f"Dim{dim2}"]
    df_var["modality"] = df_var.index.astype(str)
    
    # Split _1 and _0
    df_var_1 = df_var[df_var["modality"].str.endswith("_1")].copy()
    df_var_0 = df_var[df_var["modality"].str.endswith("_0")].copy()
    
    df_var_1["label"] = df_var_1["modality"]
    df_var_0["label"] = df_var_0["modality"]
    
    # --- Plot ---
    fig = go.Figure()
    
    # Trace 0: Individuals
    fig.add_trace(go.Scatter3d(
        x=df_ind[f"Dim{dim0}"],
        y=df_ind[f"Dim{dim1}"],
        z=df_ind[f"Dim{dim2}"],
        mode="markers",
        marker=dict(
            size=2.8,
            color="lightgrey",
            opacity=0.18
        ),
        name="Individuals",
        hoverinfo="skip"
    ))
    
    # Trace 1: Modalities _1
    fig.add_trace(go.Scatter3d(
        x=df_var_1[f"Dim{dim0}"],
        y=df_var_1[f"Dim{dim1}"],
        z=df_var_1[f"Dim{dim2}"],
        mode="markers+text",
        marker=dict(
            size=7,
            color="royalblue",
            opacity=0.95
        ),
        text=df_var_1["label"],
        textposition="top center",
        name="Modalities = 1",
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Dim 0 = %{x:.3f}<br>"
            "Dim 1 = %{y:.3f}<br>"
            "Dim 2 = %{z:.3f}"
            "<extra></extra>"
        )
    ))
    
    # Trace 2: Modalities _0
    fig.add_trace(go.Scatter3d(
        x=df_var_0[f"Dim{dim0}"],
        y=df_var_0[f"Dim{dim1}"],
        z=df_var_0[f"Dim{dim2}"],
        mode="markers+text",
        marker=dict(
            size=7,
            color="crimson",
            opacity=0.95
        ),
        text=df_var_0["label"],
        textposition="top center",
        name="Modalities = 0",
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Dim 0 = %{x:.3f}<br>"
            "Dim 1 = %{y:.3f}<br>"
            "Dim 2 = %{z:.3f}"
            "<extra></extra>"
        )
    ))
    
    # --- Buttons ---
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.02,
                y=1.08,
                showactive=True,
                buttons=[
                    dict(
                        label="Show all",
                        method="update",
                        args=[{"visible": [True, True, True]}]
                    ),
                    dict(
                        label="Hide individuals",
                        method="update",
                        args=[{"visible": [False, True, True]}]
                    ),
                ]
            )
        ]
    )
    
    fig.update_layout(
        title="3D MCA map: individuals in background and modality coordinates",
        scene=dict(
            xaxis_title=f"Dim{dim0}",
            yaxis_title=f"Dim{dim1}",
            zaxis_title=f"Dim{dim2}"
        ),
        width=1100,
        height=800
    )
    
    fig.show()
    if write:
        fig.write_html(
            f"{name}.html",
            include_plotlyjs="cdn",
            full_html=True
        )
    
