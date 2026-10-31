"""
Interactive front-end for the neutron shielding Monte Carlo simulation.

Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from neutron_sim import MATERIALS, Layer, Material, fit_attenuation, run_trials, thickness_sweep

st.set_page_config(page_title='Neutron Shielding Monte Carlo', layout='wide')

FATES = ('reflected', 'transmitted', 'absorbed')
FATE_COLORS = {'reflected': '#e8871e', 'transmitted': '#7f7f7f', 'absorbed': '#2a6fdb'}
MATERIAL_COLUMNS = {
    'name': 'Material',
    'density': 'Density (g/cm³)',
    'molar_mass': 'Molar mass (g/mol)',
    'absorption_cross_section': 'σ absorption (b)',
    'scattering_cross_section': 'σ scattering (b)',
}


def default_materials_table():
    rows = [{column: getattr(material, field) for field, column in MATERIAL_COLUMNS.items()}
            for material in MATERIALS.values()]
    return pd.DataFrame(rows)


def materials_from_table(table):
    """Build Material objects from the edited table, returning (materials, errors)."""
    materials, errors = {}, []
    for _, row in table.iterrows():
        name = row[MATERIAL_COLUMNS['name']]
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        try:
            values = {field: float(row[column]) for field, column in MATERIAL_COLUMNS.items() if field != 'name'}
            if any(np.isnan(v) for v in values.values()):
                raise ValueError(f'{name}: all properties must be filled in.')
            if name in materials:
                raise ValueError(f'{name}: duplicate material name.')
            materials[name] = Material(name, **values)
        except (TypeError, ValueError) as error:
            errors.append(str(error))
    return materials, errors


def material_key(material):
    """Hashable, cache-friendly description of a material."""
    return (material.name, material.density, material.molar_mass,
            material.absorption_cross_section, material.scattering_cross_section)


@st.cache_data(show_spinner=False, max_entries=50)
def cached_stack_run(layer_keys, n_neutrons, n_trials, seed, n_tracked):
    layers = [Layer(Material(*key), thickness) for key, thickness in layer_keys]
    return run_trials(layers, n_neutrons, n_trials, np.random.default_rng(seed), n_tracked)


@st.cache_data(show_spinner=False, max_entries=200)
def cached_sweep(material_key_, thicknesses, n_neutrons, n_trials, seed):
    return thickness_sweep(Material(*material_key_), thicknesses, n_neutrons, n_trials,
                           np.random.default_rng(seed))


# --------------------------------------------------------------------------------------
# Sidebar: global Monte Carlo settings
# --------------------------------------------------------------------------------------
st.sidebar.header('Monte Carlo settings')
n_neutrons = st.sidebar.number_input('Neutrons per trial', min_value=100, max_value=200_000,
                                     value=10_000, step=1000)
n_trials = st.sidebar.number_input('Trials (for error estimates)', min_value=2, max_value=50, value=10)
seed = st.sidebar.number_input('Random seed', min_value=0, value=0, step=1,
                               help='The same seed and settings always give the same results.')
st.sidebar.caption('Errors are standard errors of the mean over the trials.')

st.title('Neutron Shielding Monte Carlo')
st.caption('Thermal neutrons hit a stack of shielding layers at normal incidence. Each neutron is '
           'followed until it is reflected (x < 0), transmitted (beyond the last layer) or absorbed.')

tab_materials, tab_stack, tab_sweep, tab_about = st.tabs(
    ['Materials', 'Shield simulation', 'Thickness sweep', 'Model'])

# --------------------------------------------------------------------------------------
# Materials
# --------------------------------------------------------------------------------------
with tab_materials:
    st.subheader('Material library')
    st.write('Edit any property, or add a row at the bottom of the table to define a new material. '
             'Cross-sections are per formula unit (molecule or atom) for thermal neutrons.')

    if 'materials_table' not in st.session_state:
        st.session_state.materials_table = default_materials_table()
    if st.button('Reset to default materials'):
        st.session_state.materials_table = default_materials_table()
        st.session_state.pop('materials_editor', None)
        st.rerun()

    edited_table = st.data_editor(
        st.session_state.materials_table, key='materials_editor', num_rows='dynamic',
        width='stretch', hide_index=True,
        column_config={
            MATERIAL_COLUMNS['density']: st.column_config.NumberColumn(min_value=0.0, format='%.4g'),
            MATERIAL_COLUMNS['molar_mass']: st.column_config.NumberColumn(min_value=0.0, format='%.4g'),
            MATERIAL_COLUMNS['absorption_cross_section']: st.column_config.NumberColumn(min_value=0.0, format='%.4g'),
            MATERIAL_COLUMNS['scattering_cross_section']: st.column_config.NumberColumn(min_value=0.0, format='%.4g'),
        })
    materials, material_errors = materials_from_table(edited_table)
    for error in material_errors:
        st.warning(f'Ignored: {error}')

    st.subheader('Derived quantities')
    derived = pd.DataFrame([{
        'Material': m.name,
        'Σ absorption (cm⁻¹)': m.sigma_absorption,
        'Σ scattering (cm⁻¹)': m.sigma_scattering,
        'Σ total (cm⁻¹)': m.sigma_total,
        'λ absorption (cm)': m.absorption_mean_free_path,
        'λ scattering (cm)': m.scattering_mean_free_path,
        'λ total (cm)': m.mean_free_path,
        'P(absorption) per collision': m.absorption_probability,
        'Diffusion length (cm)': (1 / np.sqrt(3 * m.sigma_absorption * m.sigma_total)
                                  if m.sigma_absorption > 0 else np.inf),
    } for m in materials.values()])
    st.dataframe(derived, width='stretch', hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format='%.4g')
                                for c in derived.columns if c != 'Material'})
    st.caption('Diffusion length L = 1/√(3 Σₐ Σₜ) is the diffusion-theory estimate of the attenuation '
               'length in a thick slab (isotropic scattering, Σₐ ≪ Σₜ).')

material_names = list(materials)

# --------------------------------------------------------------------------------------
# Shield simulation
# --------------------------------------------------------------------------------------
with tab_stack:
    st.subheader('Layer stack')
    st.write('Layers are listed in the order the beam meets them. Add or remove rows to build '
             'multi-layer shields; layers of different materials use Woodcock (delta) tracking.')

    if not material_names:
        st.error('Define at least one valid material in the Materials tab.')
        st.stop()

    default_layers = pd.DataFrame({'Material': [n for n in ('Lead', 'Water') if n in materials] or material_names[:1],
                                   'Thickness (cm)': 10.0})
    layer_table = st.data_editor(
        default_layers, key='layers_editor', num_rows='dynamic', hide_index=True,
        column_config={
            'Material': st.column_config.SelectboxColumn(options=material_names, required=True),
            'Thickness (cm)': st.column_config.NumberColumn(min_value=0.001, required=True, format='%.3g'),
        })
    layer_rows = [(row['Material'], float(row['Thickness (cm)'])) for _, row in layer_table.iterrows()
                  if row['Material'] in materials and pd.notna(row['Thickness (cm)'])
                  and row['Thickness (cm)'] > 0]

    n_tracked = st.slider('Trajectories to draw', 0, 30, 5)
    stack_params = (tuple((material_key(materials[name]), thickness) for name, thickness in layer_rows),
                    int(n_neutrons), int(n_trials), int(seed), n_tracked)

    if st.button('Run simulation', type='primary', disabled=not layer_rows):
        st.session_state.stack_params = stack_params

    if 'stack_params' in st.session_state:
        if st.session_state.stack_params != stack_params:
            st.info('Settings changed since the last run. Press **Run simulation** to update.')
        params = st.session_state.stack_params
        with st.spinner('Simulating...'):
            stats = cached_stack_run(*params)
        result = stats.result
        layer_keys = params[0]
        layer_ends = np.cumsum([thickness for _, thickness in layer_keys])
        total_neutrons = stats.n_neutrons * stats.n_trials

        columns = st.columns(3)
        for column, fate in zip(columns, FATES):
            column.metric(fate.capitalize(), f'{100 * stats.mean[fate]:.2f} %')
            column.caption(f'± {100 * stats.sem[fate]:.2f} % (standard error over {stats.n_trials} trials)')
        st.caption(f'{total_neutrons:,} neutrons in total, {result.collisions:,} real collisions, '
                   f'{result.fictitious_collisions:,} fictitious collisions.')

        left, right = st.columns(2)
        with left:
            figure = go.Figure(go.Pie(labels=[f.capitalize() for f in FATES],
                                      values=[stats.mean[f] for f in FATES],
                                      marker_colors=[FATE_COLORS[f] for f in FATES], sort=False, hole=0.4))
            figure.update_layout(title='Neutron fates', height=380, margin=dict(t=50, b=10))
            st.plotly_chart(figure, width='stretch')
        with right:
            labels = [f'{i + 1}. {key[0]} ({thickness:g} cm)' for i, (key, thickness) in enumerate(layer_keys)]
            figure = go.Figure(go.Bar(x=labels, y=100 * result.absorbed_per_layer / total_neutrons,
                                      marker_color=FATE_COLORS['absorbed']))
            figure.update_layout(title='Absorbed in each layer', yaxis_title='% of all neutrons',
                                 height=380, margin=dict(t=50, b=10))
            st.plotly_chart(figure, width='stretch')

        if len(result.absorption_depths):
            figure = go.Figure(go.Histogram(x=result.absorption_depths, nbinsx=60,
                                            marker_color=FATE_COLORS['absorbed']))
            for end in layer_ends[:-1]:
                figure.add_vline(x=end, line_dash='dash', line_color='black')
            figure.update_layout(title='Where neutrons are absorbed', xaxis_title='Depth x (cm)',
                                 yaxis_title='Neutrons', yaxis_type='log', height=380, margin=dict(t=50, b=10))
            st.plotly_chart(figure, width='stretch')

        if result.paths:
            figure = go.Figure()
            all_points = np.vstack(result.paths)
            extent = max(np.abs(all_points[:, 1:]).max(), 1.0)
            for x in np.concatenate([[0], layer_ends]):
                figure.add_trace(go.Mesh3d(x=[x] * 4, y=[-extent, extent, extent, -extent],
                                           z=[-extent, -extent, extent, extent], i=[0, 0], j=[1, 2], k=[2, 3],
                                           opacity=0.15, color='gray', hoverinfo='skip', showlegend=False))
            for index, path in enumerate(result.paths):
                figure.add_trace(go.Scatter3d(x=path[:, 0], y=path[:, 1], z=path[:, 2], mode='lines+markers',
                                              marker_size=2, line_width=3, name=f'Neutron {index + 1}'))
            figure.update_layout(title='Neutron trajectories (planes mark layer boundaries)', height=600,
                                 scene=dict(xaxis_title='x (cm)', yaxis_title='y (cm)', zaxis_title='z (cm)',
                                            aspectmode='data'),
                                 margin=dict(t=50, b=10))
            st.plotly_chart(figure, width='stretch')

# --------------------------------------------------------------------------------------
# Thickness sweep
# --------------------------------------------------------------------------------------
with tab_sweep:
    st.subheader('Transmission against thickness')
    st.write('Simulates single-material slabs over a range of thicknesses and fits '
             'ln(T) = −L/λ + c to find the attenuation length λ.')

    selected = st.multiselect('Materials', material_names,
                              default=[n for n in ('Water', 'Lead', 'Graphite', 'Beryllium') if n in materials])
    c1, c2, c3 = st.columns(3)
    thickness_min = c1.number_input('Minimum thickness (cm)', min_value=0.01, value=1.0)
    thickness_max = c2.number_input('Maximum thickness (cm)', min_value=0.02, value=20.0)
    n_points = c3.number_input('Number of thicknesses', min_value=3, max_value=60, value=20)
    sweep_params = (tuple(material_key(materials[name]) for name in selected),
                    tuple(np.linspace(thickness_min, thickness_max, int(n_points))),
                    int(n_neutrons), int(n_trials), int(seed))

    if st.button('Run sweep', type='primary', disabled=not selected or thickness_max <= thickness_min):
        st.session_state.sweep_params = sweep_params

    if 'sweep_params' in st.session_state:
        if st.session_state.sweep_params != sweep_params:
            st.info('Settings changed since the last run. Press **Run sweep** to update.')
        keys, thicknesses, *run_settings = st.session_state.sweep_params

        progress = st.progress(0.0, text='Simulating...')
        sweeps = {}
        for i, key in enumerate(keys):
            progress.progress(i / len(keys), text=f'Simulating {key[0]}...')
            sweeps[key[0]] = cached_sweep(key, thicknesses, *run_settings)
        progress.empty()

        fit_range = st.slider('Thickness range used in the fit (cm)', float(thicknesses[0]), float(thicknesses[-1]),
                              (float(thicknesses[0]), float(thicknesses[-1])),
                              help='Transmission is only exponential in thick slabs; start the fit '
                                   'further in to exclude the build-up region at small thicknesses.')

        figure = make_subplots(rows=1, cols=3, subplot_titles=[f.capitalize() for f in FATES],
                               shared_yaxes=True)
        palette = ['#2a6fdb', '#e8871e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf']
        for i, (name, sweep) in enumerate(sweeps.items()):
            color = palette[i % len(palette)]
            for col, fate in enumerate(FATES, start=1):
                figure.add_trace(go.Scatter(x=sweep['thickness'], y=sweep[fate],
                                            error_y=dict(array=sweep[fate + '_sem']), mode='lines+markers',
                                            name=name, legendgroup=name, showlegend=col == 1,
                                            line_color=color), row=1, col=col)
        figure.update_xaxes(title_text='Thickness (cm)')
        figure.update_yaxes(title_text='Fraction of neutrons', row=1, col=1)
        figure.update_layout(height=420, margin=dict(t=50, b=10))
        st.plotly_chart(figure, width='stretch')

        figure = go.Figure()
        fit_rows = []
        for i, (name, sweep) in enumerate(sweeps.items()):
            color = palette[i % len(palette)]
            in_range = (sweep['thickness'] >= fit_range[0] - 1e-9) & (sweep['thickness'] <= fit_range[1] + 1e-9)
            fit = fit_attenuation(sweep['thickness'][in_range], sweep['transmitted'][in_range],
                                  sweep['transmitted_sem'][in_range])
            figure.add_trace(go.Scatter(x=sweep['thickness'], y=sweep['transmitted'],
                                        error_y=dict(array=sweep['transmitted_sem']), mode='markers',
                                        name=name, legendgroup=name, marker_color=color))
            material = Material(*next(key for key in keys if key[0] == name))
            row = {'Material': name, 'Attenuation length λ (cm)': np.nan, 'Error (cm)': np.nan,
                   'Reduced χ²': np.nan,
                   'Diffusion length (cm)': (1 / np.sqrt(3 * material.sigma_absorption * material.sigma_total)
                                             if material.sigma_absorption > 0 else np.inf)}
            if fit is not None:
                x = np.linspace(fit_range[0], fit_range[1], 100)
                figure.add_trace(go.Scatter(x=x, y=np.exp(fit.intercept + fit.slope * x), mode='lines',
                                            name=f'{name} fit', legendgroup=name, line_color=color,
                                            showlegend=False))
                row.update({'Attenuation length λ (cm)': fit.attenuation_length,
                            'Error (cm)': fit.attenuation_length_error, 'Reduced χ²': fit.reduced_chi_squared})
            fit_rows.append(row)
        figure.update_layout(title='Transmitted fraction (log scale) with exponential fits',
                             xaxis_title='Thickness (cm)', yaxis_title='Transmitted fraction',
                             yaxis_type='log', height=450, margin=dict(t=50, b=10))
        st.plotly_chart(figure, width='stretch')

        st.dataframe(pd.DataFrame(fit_rows), hide_index=True, width='stretch',
                     column_config={c: st.column_config.NumberColumn(format='%.4g')
                                    for c in ('Attenuation length λ (cm)', 'Error (cm)', 'Reduced χ²',
                                              'Diffusion length (cm)')})
        st.caption('No fit is shown when fewer than three thicknesses in the range have any transmitted '
                   'neutrons. A large reduced χ² means transmission is not purely exponential over the '
                   'fit range.')

        export = pd.concat([pd.DataFrame({'material': name, **sweep}) for name, sweep in sweeps.items()])
        st.download_button('Download sweep results (CSV)', export.to_csv(index=False),
                           file_name='thickness_sweep.csv', mime='text/csv')

# --------------------------------------------------------------------------------------
# Model description
# --------------------------------------------------------------------------------------
with tab_about:
    st.markdown(r'''
### Physical model
- A mono-energetic beam of **thermal neutrons** enters the shield at $x = 0$, moving in $+x$.
- Layers are **infinite slabs** in $y$ and $z$, stacked along $x$.
- The distance to the next collision is exponential with mean free path $\lambda = 1/\Sigma_t$, where
  $\Sigma = \rho N_A \sigma / M$ is the macroscopic cross-section ($\sigma$ in barns, 1 b = $10^{-24}$ cm²).
- At a collision the neutron is **absorbed** with probability $\Sigma_a/\Sigma_t$, otherwise it
  **scatters isotropically** with no energy loss (one-speed model).
- A neutron is **reflected** when $x < 0$ and **transmitted** when it passes the last layer.

### Woodcock (delta) tracking
Flight distances are sampled with the largest cross-section in the stack, $\Sigma_{max}$. At a collision
point in a layer with $\Sigma_t$, the collision is real with probability $\Sigma_t / \Sigma_{max}$; otherwise it is
*fictitious* and the neutron continues in the same direction. This gives exact results without having to
find where each flight crosses a layer boundary. For a single layer every collision is real.

### Limitations
- No energy dependence: thermal cross-sections are used throughout, so moderation and
  energy-dependent absorption (for example 1/v absorbers) are not modelled.
- Scattering is isotropic in the laboratory frame.
- Cross-sections for hydrogenous materials depend on chemical binding; the library values are approximate.
''')
