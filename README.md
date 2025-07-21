# dissipative\_cooling

A minimal, reproducible playground for **dissipative (algorithmic) cooling** of a quantum register using an ancilla qubit that is periodically reset.  The repository shows how to remove entropy from a target system by coupling it to a controllable dissipative element built from hardware‑native $T_1$ relaxation.

---

## 1  Concept in a Nutshell

> *Pump heat into a bucket you can empty.*
>
> 1. **System register** $\mathcal S$: the qubits (spins) whose energy you want to lower.
> 2. **Ancilla / bath qubit** $a$: a single qubit that we can *reset* to $|0\rangle$ at will.
> 3. **System–bath coupling** allows energy to flow from $\mathcal S$ into $a$.
> 4. **Frequent resets** erase the entropy that has accumulated in $a$, therefore enforcing a *directional* flow of heat out of $\mathcal S$.

Repeated coupling‑and‑reset cycles drive $\mathcal S$ toward its ground state, provided the reset rate is faster than intrinsic heating noise.

---

## 2  Hamiltonian

We consider a separable Hamiltonian

$$
H = H_{\text{sys}} + H_{a} + H_{\text{int}},
$$

where

* **System**: $H_{\text{sys}} = \sum_{i<j} J_{ij} \sigma_i^z \sigma_j^z + \sum_i h_i \sigma_i^x$
* **Ancilla**: $H_a = \frac{\omega_a}{2}\,\sigma_a^z$
* **Interaction**: a resonant, energy‑preserving exchange (e.g. an XX‑coupling)

  $$H_{\text{int}} = g \left( \sigma_1^x \sigma_a^x + \sigma_1^y \sigma_a^y \right),$$

  that swaps excitations between the *edge* system qubit 1 and the ancilla.

The choice of the edge qubit is arbitrary—any controllable qubit of $\mathcal S$ can be wired to the ancilla.

---

## 3  Reset‑Based Dissipation

A *hardware reset* (or active measurement + preparation) applies an effective **lowering operator**

$$
\mathcal R(\rho) = |0\rangle_a\!\langle0| \otimes \operatorname{Tr}_a(\rho),
$$

which is equivalent, at the master‑equation level, to the dissipator

$$
\mathcal D_a[\rho] = \gamma \,\bigl( \sigma^-_a \,\rho\, \sigma^+_a -\tfrac12\{\sigma^+_a \sigma^-_a,\rho\} \bigr).
$$

Integrating the unitary swap $e^{-i H_{\text{int}} \tau}$ with a fast reset gives the stroboscopic map

$$
\rho_{n+1} = \mathcal R\bigl[ U_{\text{int}}\,\rho_n\,U_{\text{int}}^\dagger \bigr],
$$

which, in the limit of small cycle time, converges to a Lindblad evolution where the ancilla acts as a *zero‑temperature bath* for the coupled system qubit. Under ergodic conditions, the entire register cools toward the ground state of $H_{\text{sys}}$.

---
