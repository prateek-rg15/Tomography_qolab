# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM, Transmon
from quam_libs.macros import qua_declaration, active_reset, readout_state
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import time


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = ["q4"]
    use_state_discrimination: bool = True
    num_averages: int = 500          # averages per basis
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    # prepare |0>, |1>, |+>, or |i>
    prepared_state: Literal["zero", "one", "plus", "plus_i"] = "zero"
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False


description = """Single-Qubit State Tomography:"""

node = QualibrationNode(
    name="10b_Single_Qubit_State_Tomography",
    description=description,
    parameters=Parameters(),
)



def apply_pauli_basis_rotation(qubit: Transmon, pauli_id: int):

    with switch_(pauli_id, unsafe=True):
        with case_(0):
            qubit.xy.play("y90")
        with case_(1):
            qubit.xy.play("-x90")
        with case_(2):
            pass


# %% {Initialize_QuAM_and_QOP}
u = unit(coerce_to_integer=True)
machine = QuAM.load()
config = machine.generate_config()

if node.parameters.load_data_id is None and not node.parameters.simulate:
    qmm = machine.connect()

# Select qubits
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)
assert num_qubits == 1, "This tomography node is written for a single qubit."
qubit: Transmon = qubits[0]

n_avg = node.parameters.num_averages
flux_point = node.parameters.flux_point_joint_or_independent
reset_type = node.parameters.reset_type_thermal_or_active
use_state_disc = node.parameters.use_state_discrimination
prepared_state = node.parameters.prepared_state  # "zero", "one", "plus", "plus_i"


basis_labels = np.array(["X", "Y", "Z"])
num_bases = len(basis_labels)



with program() as state_tomography:

    snapshot = declare(int)
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)

    state = [declare(int) for _ in range(num_qubits)]
    state_st = [declare_stream() for _ in range(num_qubits)]
    snapshot_st = declare_stream()


    machine.set_all_fluxes(flux_point=flux_point, target=qubit)


    with for_(snapshot, 0, snapshot < num_bases, snapshot + 1):

        save(snapshot, snapshot_st)

        with for_(n, 0, n < n_avg, n + 1):

            if reset_type == "active":
                active_reset(qubit, "readout")
            else:
                qubit.resonator.wait(qubit.thermalization_time * u.ns)

            qubit.align()


            if prepared_state == "one":
                qubit.xy.play("x180")
            elif prepared_state == "plus":
                qubit.xy.play("-y90")
            elif prepared_state == "plus_i":
                qubit.xy.play("x90")

            qubit.align()

            # Apply basis rotation
            apply_pauli_basis_rotation(qubit, snapshot)
            qubit.align()


            if use_state_disc:
                readout_state(qubit, state[0])
                save(state[0], state_st[0])
            else:
                qubit.resonator.measure("readout", qua_vars=(I[0], Q[0]))
                save(I[0], I_st[0])

            # Relaxation
            qubit.resonator.wait(qubit.thermalization_time * u.ns)
            qubit.align()

    with stream_processing():
        snapshot_st.save("snapshot")
        if use_state_disc:
            state_st[0].buffer(n_avg).map(FUNCTIONS.average()).buffer(num_bases).save("state1")
        else:
            I_st[0].buffer(n_avg).map(FUNCTIONS.average()).buffer(num_bases).save("I1")
            Q_st[0].buffer(n_avg).map(FUNCTIONS.average()).buffer(num_bases).save("Q1")



if node.parameters.simulate:
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns)
    job = qmm.simulate(config, state_tomography, simulation_config)
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()), 1, i + 1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

elif node.parameters.load_data_id is None:
    node.results = {}
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        start_time = time.time()
        job = qm.execute(state_tomography)
        job.result_handles.wait_for_all_values()
        end_time = time.time()
        node.results["execution_time_seconds"] = end_time - start_time
else:
    job = None



if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        ds = fetch_results_as_xarray(
            job.result_handles,
            qubits,
            {"snapshot": np.arange(num_bases)},
        )
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]

    node.results["ds"] = ds

    if use_state_disc:
        if "state" in ds.data_vars:
            da_state = ds["state"]
        else:
            state_vars = [k for k in ds.data_vars if k.startswith("state")]
            if not state_vars:
                raise RuntimeError(
                    f"No 'state' found. Data variables: {list(ds.data_vars)}"
                )
            da_state = ds[state_vars[0]]

        if "qubit" in da_state.dims:
            da_state = da_state.isel(qubit=0)

        da_state = da_state.assign_coords(snapshot=("snapshot", basis_labels))
        da_state = da_state.rename(snapshot="basis")

        p1 = da_state
        px = 1.0 - 2.0 * p1.sel(basis="X")
        py = 1.0 - 2.0 * p1.sel(basis="Y")
        pz = 1.0 - 2.0 * p1.sel(basis="Z")

        rx = float(px.values)
        ry = float(py.values)
        rz = float(pz.values)

        F0 = (1.0 + rz) / 2.0
        F1 = (1.0 - rz) / 2.0
        F_plus = (1.0 + rx) / 2.0
        F_i = (1.0 + ry) / 2.0

        node.results["bloch_vector"] = {"rx": rx, "ry": ry, "rz": rz}
        node.results["fidelity_to_0"] = F0
        node.results["fidelity_to_1"] = F1
        node.results["fidelity_to_plus"] = F_plus
        node.results["fidelity_to_i"] = F_i

        fig, axes = plt.subplots(1, 2, figsize=(9, 4))

        ax = axes[0]
        ax.bar(["X", "Y", "Z"], [rx, ry, rz])
        ax.set_ylim(-1.1, 1.1)
        ax.set_ylabel("<σ>")
        ax.set_title(f"{qubit.name} Bloch components ({prepared_state} state)")

        ax = axes[1]
        if prepared_state == "zero":
            ax.bar(["F(|0>)"], [F0])
            ax.set_title("Tomography fidelity to |0>")
        elif prepared_state == "one":
            ax.bar(["F(|1>)"], [F1])
            ax.set_title("Tomography fidelity to |1>")
        elif prepared_state == "plus":
            ax.bar(["F(|+>)"], [F_plus])
            ax.set_title("Tomography fidelity to |+>")
        else:
            ax.bar(["F(|i>)"], [F_i])
            ax.set_title("Tomography fidelity to |i>")

        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Fidelity")
        plt.tight_layout()
        node.results["figure"] = plt.gcf()

    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()
