import qutip as qt
import numpy as np
import warnings
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from dataclasses import dataclass
import matplotlib.pyplot as plt # Import matplotlib for plotting

# Suppress harmless warnings from QuTiP and numpy
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

@dataclass
class OptimizationBounds:
    """
    Defines the search space bounds for the optimization parameters.
    """
    fx: tuple = (-2.0, 2.0)
    fy: tuple = (-2.0, 2.0)
    fz: tuple = (-2.0, 2.0)
    delta_bath: tuple = None  # This will be set dynamically after gap estimation
    gamma: tuple = (0.1, 10.0)

class ED:
    """
    Exact Diagonalization helper: generates spin operators for n qubits.
    This class provides a convenient way to create Pauli matrices (sigma_x, sigma_y, sigma_z)
    and lowering operators (sigma_minus) for a multi-qubit system, correctly tensored
    with identity operators for other qubits.
    """
    def __init__(self, n_qubits: int):
        self.n = n_qubits
        id2 = qt.identity(2) # 2x2 identity matrix for a single qubit

        # List of sigma_x operators for each qubit
        # e.g., for n=3, sx[0] = sigmax x id2 x id2, sx[1] = id2 x sigmax x id2, etc.
        self.sx = [qt.tensor(*[id2]*i + [qt.sigmax()] + [id2]*(self.n-i-1)) for i in range(self.n)]
        self.sy = [qt.tensor(*[id2]*i + [qt.sigmay()] + [id2]*(self.n-i-1)) for i in range(self.n)]
        self.sz = [qt.tensor(*[id2]*i + [qt.sigmaz()] + [id2]*(self.n-i-1)) for i in range(self.n)]
        self.sm = [qt.tensor(*[id2]*i + [qt.sigmam()] + [id2]*(self.n-i-1)) for i in range(self.n)]

class HamiltonianBuilder:
    """
    Builds the system, bath, and interaction Hamiltonians.
    It constructs the total Hamiltonian based on the given parameters.
    """
    def __init__(self, ed: ED, J: float, hx: float, n_sys: int):
        self.ed = ed # ED object for operator generation
        self.J = J # Coupling strength for system qubits
        self.hx = hx # Magnetic field strength for system qubits
        self.n_sys = n_sys # Number of system qubits

    def system_hamiltonian(self) -> qt.Qobj:
        """
        Constructs the Hamiltonian for the system qubits (e.g., an Ising chain).
        """
        H = 0
        # Sum over nearest-neighbor interactions and local magnetic fields
        for i in range(self.n_sys - 1):
            H += self.J * self.ed.sx[i] * self.ed.sx[i+1] # Ising coupling
            H += self.hx * (self.ed.sz[i] + self.ed.sz[i+1]) # Local magnetic field
        return H

    def total_hamiltonian(self, fx: float, fy: float, fz: float, delta: float) -> qt.Qobj:
        """
        Constructs the total Hamiltonian, including system, bath, and interaction terms.
        The bath is assumed to be the (n_sys)-th qubit.
        """
        H_sys = self.system_hamiltonian() # System Hamiltonian
        idx = self.n_sys # Index of the bath qubit (n_sys is the (n_sys+1)-th qubit)

        H_bath = delta * self.ed.sz[idx] # Bath Hamiltonian (single qubit with detuning delta)

        # Interaction Hamiltonian between the last system qubit (idx-1) and the bath qubit (idx)
        H_int  = fx * self.ed.sx[idx-1] * self.ed.sx[idx]
        H_int += fy * self.ed.sy[idx-1] * self.ed.sy[idx]
        H_int += fz * self.ed.sz[idx-1] * self.ed.sz[idx]

        return H_sys + H_bath + H_int

class SympatheticCoolingSimulator:
    """
    Encapsulates the objective function for optimization, runs the optimization,
    and performs the final simulation and plotting.
    """
    def __init__(self, builder: HamiltonianBuilder, bounds: OptimizationBounds,
                 n_sys: int, n_bath: int):
        self.builder = builder # HamiltonianBuilder object
        self.bounds = bounds # OptimizationBounds object
        self.n_sys = n_sys # Number of system qubits
        self.n_bath = n_bath # Total number of qubits (system + bath)

        # Initialize delta_bath bounds with a general scale-based approach as a fallback.
        # This handles cases where n_sys is 0 or 1 (where system_hamiltonian is trivial),
        # or when eigenenergy calculation fails for larger systems.
        scale = max(abs(builder.J), abs(builder.hx))
        if scale == 0: # Ensure scale is not zero if both J and hx are zero
            scale = 1.0 # Default scale if no coupling/field
        self.bounds.delta_bath = (-1.5*scale, 1.5 * scale)

        # If the system has between 2 and 4 qubits (inclusive), attempt to calculate the gap from eigenenergies.
        # For n_sys < 2 or n_sys > 4, the system Hamiltonian is either trivial or too large for ED.
        if 2 <= n_sys <= 4: # Modified condition as per your request
            H_sys = builder.system_hamiltonian()
            try:
                ee = H_sys.eigenenergies()
                ee.sort()
                gap = ee[1] - ee[0] # Energy gap between the two lowest energy states
                lb = max(0.01, 0.5 * gap) # Lower bound for delta_bath
                ub = max(lb + 0.1, 1.5 * gap) # Upper bound for delta_bath
                self.bounds.delta_bath = (lb, ub)
            except Exception as e:
                # If eigenenergy calculation fails (e.g., due to numerical instability),
                # we fall back to the previously set scale-based bounds.
                print(f"Warning: Could not calculate system gap for N_sys={n_sys} using eigenenergies. Falling back to scaled delta_bath bounds. Error: {e}")


        # Define the search space for Bayesian optimization using skopt.Real
        self.space = [
            Real(*self.bounds.fx, name='fx'),
            Real(*self.bounds.fy, name='fy'),
            Real(*self.bounds.fz, name='fz'),
            Real(*self.bounds.delta_bath, name='delta_bath'),
            Real(*self.bounds.gamma, name='gamma_ancilla_decay')
        ]
        # Bind the objective function with named arguments for skopt
        self.objective = use_named_args(self.space)(self._objective)

    def _objective(self, fx, fy, fz, delta_bath, gamma_ancilla_decay):
        """
        The objective function to be minimized by the optimizer.
        It simulates the system's evolution and returns the final energy of the system.
        A lower energy indicates better cooling.
        """
        # Ensure gamma_ancilla_decay is positive to avoid issues with sqrt
        if gamma_ancilla_decay <= 0:
            return 1e10 # Return a very high cost for invalid gamma

        # Build the total Hamiltonian with current parameters
        H_tot = self.builder.total_hamiltonian(fx, fy, fz, delta_bath)
        # Define collapse operator for the bath qubit's decay
        c_ops = [np.sqrt(gamma_ancilla_decay) * self.builder.ed.sm[self.n_sys]]

        # Initial state: all system qubits in excited state |1>, bath qubit in ground state |0>
        # qt.basis(2,1) is |1>, qt.basis(2,0) is |0>
        psi0 = qt.tensor(*([qt.basis(2,1)]*self.n_sys + [qt.basis(2,0)]))

        # Time list for the simulation (shorter for optimization runs)
        tlist = np.arange(0, 50, 0.5)
        # Operators to expect (measure) during the simulation
        e_ops = [self.builder.system_hamiltonian()] # We want to track the system's energy

        # QuTiP options for mesolve
        opts  = qt.Options(nsteps=100000, store_states=False, store_final_state=True)

        try:
            # Run the master equation solver
            res = qt.mesolve(H_tot, psi0, tlist, c_ops, e_ops=e_ops, options=opts)
            # Return the final energy of the system (last value in expect[0])
            return res.expect[0][-1]
        except Exception as e:
            # If simulation fails, return a high cost
            print(f"Simulation failed with error: {e}")
            return 1e10

    def run_optimization(self, n_calls=50, seed=42, verbose=True, x0=None):
        """
        Runs the Bayesian optimization using gp_minimize.
        Args:
            n_calls (int): The number of calls to the objective function.
            seed (int): Random seed for reproducibility.
            verbose (bool): Whether to print optimization progress.
            x0 (list, optional): Initial points for the optimization. Defaults to None.
        Returns:
            OptimizeResult: The result object from gp_minimize.
        """
        print(f"Starting optimization for N_sys={self.n_sys} with {n_calls} calls.")
        if x0:
            print(f"Using initial points: {x0}")
            # When x0 is provided, we need to ensure n_calls is sufficient.
            # skopt requires n_calls >= len(x0) + n_initial_points (default 10)
            # or n_calls >= len(x0) + some minimum (e.g., 1) if n_initial_points=0.
            # The error message suggests n_initial_points is effectively 6 when x0 is present.
            # To allow for 10 *new* iterations after x0, we set n_calls to len(x0) + 10.
            # We also explicitly set n_initial_points=0 to avoid skopt adding extra random points.
            return gp_minimize(self.objective, self.space,
                               n_calls=n_calls, random_state=seed,
                               verbose=verbose, x0=x0, n_initial_points=0)
        else:
            # If no initial points (x0) are provided, n_initial_points defaults to 10
            # (or n_random_starts), so n_calls=10 is fine for the first run.
            return gp_minimize(self.objective, self.space,
                               n_calls=n_calls, random_state=seed,
                               verbose=verbose, x0=x0)


    def final_run_and_plot(self, res):
        """
        Performs a final, longer simulation with the optimized parameters
        and plots the system energy and ancilla population over time.
        Args:
            res (OptimizeResult): The result object from the optimization.
        """
        # Extract optimized parameters
        fx, fy, fz, delta_bath, gamma_ancilla_decay = res.x
        print(f"\nFinal simulation with optimized parameters:")
        print(f"  fx={fx:.4f}, fy={fy:.4f}, fz={fz:.4f}, delta_bath={delta_bath:.4f}, gamma_ancilla_decay={gamma_ancilla_decay:.4f}")
        print(f"  Final optimized cost (system energy): {res.fun:.4f}")

        # Rebuild Hamiltonian and collapse operators with optimized values
        H_tot = self.builder.total_hamiltonian(fx, fy, fz, delta_bath)
        c_ops = [np.sqrt(gamma_ancilla_decay) * self.builder.ed.sm[self.n_sys]]
        psi0 = qt.tensor(*([qt.basis(2,1)]*self.n_sys + [qt.basis(2,0)]))
        H_sys = self.builder.system_hamiltonian()
        ee = H_sys.eigenenergies()
        print(f"  Eigenenergies: {ee}")
        # print(H_sys)
        # Longer time list for detailed plotting
        tlist = np.arange(0, 150, 0.1)
        # Operators to track: system energy and bath qubit's population (sigma_z)
        e_ops = [self.builder.system_hamiltonian(), self.builder.ed.sz[self.n_sys]]
        opts  = qt.Options(nsteps=500000) # Increased steps for longer simulation

        # Run the final simulation
        resf  = qt.mesolve(H_tot, psi0, tlist, c_ops, e_ops=e_ops, options=opts)

        # Plotting results
        fig, axes = plt.subplots(2,1, figsize=(10,8), sharex=True)

        # Plot system energy
        axes[0].plot(resf.times, resf.expect[0], label='System Energy')
        axes[0].axhline(resf.expect[0][-1], linestyle='--', color='r', label=f'Final Energy: {resf.expect[0][-1]:.4f}')
        axes[0].axhline(ee[0], linestyle='-', lw=1.5, color='b', label=f'Ground state Energy: {ee[0]:.4f}')
        axes[0].set_ylabel('Energy')
        axes[0].set_title(f'Sympathetic Cooling Simulation (N_sys={self.n_sys}), \n fx={fx:.4f}, fy={fy:.4f}, fz={fz:.4f}, delta_bath={delta_bath:.4f}, gamma_ancilla_decay={gamma_ancilla_decay:.4f}')
        axes[0].legend()
        axes[0].grid(True, linestyle=':', alpha=0.7)

        # Plot ancilla (bath qubit) population
        # (sigma_z expectation value + 1) / 2 converts to population of |1> state
        pop = (resf.expect[1] + 1)/2
        axes[1].plot(resf.times, pop, label='Ancilla Population')
        axes[1].set_ylabel('Population of Ancilla |1>')
        axes[1].set_xlabel('Time')
        axes[1].legend()
        axes[1].grid(True, linestyle=':', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f"symathetic_cooling_N_{self.n_sys}_J_{self.builder.J}_h_{self.builder.hx}.png")
        plt.show()

if __name__ == '__main__':
    # Define a list of system sizes to simulate sequentially
    system_sizes = [2, 4, 6, 8, 10] # Example: start with 2 qubits, then 4, then 6

    J_val = -1.0 # Coupling strength for system qubits
    hx_val = 0.2 # Magnetic field strength for system qubits

    # Initialize initial_params to None for the first optimization run
    # This will allow gp_minimize to start without prior knowledge for the smallest system
    initial_params = None

    # Loop through each system size
    for N_sys in system_sizes:
        N_bath = N_sys + 1 # Total qubits = system qubits + 1 bath qubit

        print(f"\n--- Running optimization for N_sys = {N_sys} ---")

        # Setup ED, HamiltonianBuilder, and OptimizationBounds for the current system size
        ed = ED(N_bath)
        builder = HamiltonianBuilder(ed, J_val, hx_val, n_sys=N_sys)
        bounds = OptimizationBounds() # Create new bounds object for each system size
        sim = SympatheticCoolingSimulator(builder, bounds, N_sys, N_bath)

        # Determine n_calls for the current run
        # If initial_params are provided, we want 10 *new* iterations
        # plus the number of initial points.
        if initial_params is None:
            n_calls_current = 10 # For the first run, 10 random calls
        else:
            n_calls_current = len(initial_params) + 30 # 5 initial points + 10 new iterations = 15 total calls

        # Run optimization with a reduced number of calls (10 new iterations)
        # Pass initial_params if available from a previous run
        result = sim.run_optimization(n_calls=n_calls_current, x0=initial_params)

        print(f"Best parameters for N_sys={N_sys}: {result.x}")
        print(f"Best cost for N_sys={N_sys}: {result.fun}")

        # Update initial_params with the best parameters found in the current run
        # These will be used as starting points for the next, larger system
        initial_params = result.x

        # Perform final simulation and plot for the current system size
        sim.final_run_and_plot(result)

    print("\n--- All optimizations complete ---")
