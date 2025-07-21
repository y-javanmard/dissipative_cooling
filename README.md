# dissipative\_cooling

A minimal demo of **dissipative cooling** using a resettable ancilla qubit.

---

## 1. Idea

Couple a quantum system $S$ to an ancilla qubit $a$. Repeatedly:

* Swap energy from $S$ into $a$.
* Reset $a$ to its ground state.

This removes entropy from $S$, driving it toward its ground state.

---

## 2. Hamiltonian

We consider:

$$
H = H_{\text{sys}} + H_{a} + H_{\text{int}},
$$

where:

* $H_{\text{sys}} = \sum_{i<j} J_{ij} \sigma_i^z \sigma_j^z + \sum_i h_i \sigma_i^x$
* $H_a = \frac{\omega_a}{2} \sigma^z_a$
* $H_{\text{int}} = g (\sigma^-_1 \sigma^+_a + \text{h.c.})$

The ancilla couples to one site (e.g., spin 1) in the system.

---

## 3. Reset

After interaction, reset the ancilla:

$$
\mathcal{R}(\rho) = |0\rangle\langle 0| \otimes \operatorname{Tr}_a(\rho)
$$

This acts like a cold bath, enforcing directionality in energy flow.

Repeated interaction + reset steps realize dissipative evolution toward the ground state of $H_{\text{sys}}$.
