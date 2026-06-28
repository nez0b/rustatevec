### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ 01af6d1f-f4ca-4cc9-8fb3-61f1b0242848
using LinearAlgebra

# ╔═╡ 57ed8099-7ede-4bb8-9a28-2945083baacf
using Chairmarks

# ╔═╡ 228b7313-ec14-4f2d-8c69-f9f6de909a10
using Profile

# ╔═╡ d1b99d47-a25a-4e24-a3fa-a2b3023cd194
using AliasTables

# ╔═╡ 42b097f0-4c69-11f0-39d8-55affea6301b
md"
# Introduction to state vector simulation

In this lesson, we're going to implement a simple state vector simulator and steadily made it less simple (and more performant) as a way to examine some performance guidelines for Julia and numerical physics.

First, let's remind ourselves of some basic facts about state vectors.
"

# ╔═╡ 2ee77de2-ea9b-4d1e-9af7-7acfe0b6af10
md"
## State vectors, gates, and observables
"

# ╔═╡ f2ae4404-8f42-4078-8079-d46c4bacabfe
md"
### State vector basics

A *state vector* is a length-$D^N$ complex vector representing probability amplitudes for a quantum system of $N$ particles, each with $D$ possible states. So, for *qubits*, which have 2 possible states ($|0\rangle$ and $|1\rangle$), the state vector has $2^N$ elements, and for *qutrits*, which have 3 possible states, the state vector would have $3^N$ elements.

**In this lecture we'll focus entirely on the qubit case, as it's most common.**

For just a single qubit, we have a 2-element vector:

```math
|\psi_1\rangle = \begin{bmatrix} \psi_0 \\ \psi_1 \end{bmatrix}
```

Where this corresponds to 

```math
|\psi_1\rangle = \psi_0 | 0 \rangle + \psi_1 | 1 \rangle
```

Suppose we have two qubits, and we want to represent the state vector of both of them. In that case we could write:

```math
|\psi_2\rangle = \begin{bmatrix} \psi_0 \\ \psi_1 \\ \psi_2 \\ \psi_3 \end{bmatrix}
```

Here, $\psi_0$ is obviously the coefficient of $|00\rangle$ (all down) and $\psi_3$ is the coefficient of $|11\rangle$ (all up). But what about $\psi_1$ and $\psi_2$? Often in textbooks we'll see 

```math
|\psi_2\rangle = \psi_0 |00\rangle + \psi_1 |01\rangle + \psi_2 |10\rangle + \psi_3 |11\rangle
```

and say that qubit 0 is the \"lefttmost\" qubit and qubit 1 is the \"rightmost\" one. But this is a **choice**, and its one that we encounter in computer science as well, when deciding how to encode integers -- in that context it's called [endianness](https://en.wikipedia.org/wiki/Endianness). For now, we only need to remember that we are making a choice here, but we'll return to this point later.

#### Making bigger state vectors from smaller ones

Any state vector, which represents a pure state, is representable as a linear combination of [Kronecker products](https://en.wikipedia.org/wiki/Kronecker_product) of unique eigenstates on each site/qubit. Normally, of course, we just write

```math
|\psi\rangle = \psi_0 |000\rangle + \psi_7|111\rangle
```
rather than
```math
|\psi\rangle = \psi_0 |0\rangle\otimes|0\rangle\otimes|0\rangle + \psi_7 |1\rangle\otimes|1\rangle\otimes|1\rangle
```

But it's worthwhile for us to remember that we can quickly combine state vectors with the Kronecker product.
"

# ╔═╡ 66e92550-586d-4726-9478-7097ae294ec6
md"
### Applying gates to a state vector (naive approach)

Let's now try to apply some gates to our qubits so we can simulate a quantum computer.

#### An important aside

State vectors represent **pure** states, so for now we will neglect considering noise and avoid having to worry about mixed states and density matrices. However, we'll see later that a lot of what we develop here can be easily re-used for the density matrix case!

#### What's a quantum gate?

In order to perform useful computations, we apply quantum gates which represent an analogue of classical [logic gates](https://en.wikipedia.org/wiki/Logic_gate) such as `NAND`, `OR`, etc. Later this week we'll discuss simulation with a subset of gates known as Clifford gates which can be done extremely efficiently, but here we consider generic quantum gates.

A gate is an $m$-qubit *unitary operator* on $1 \leq m \leq N$ qubits. For our purposes, that means each gate can be represented as a $2^m \times 2^m$ matrix of complex numbers.

Some common gates:
- `X`, `Y`, and `Z` - the Pauli operators
- `H` - the Hadamard gate
- `RX`, `RY`, `RZ` - rotation gates along each axis
- `CNOT` - controlled `X` operator
- `CCNOT` - doubly-controlled `X`
- `SWAP` - swap state of 2 qubits
- `XX`, `YY`, `ZZ` - two qubit rotations about an axis

... and many more! For example, check out the [list of all gates](https://docs.quantum.ibm.com/api/qiskit/circuit_library) that IBM Qiskit supports.

Since any gate is a unitary operator, we can apply it as we learned in Quantum 101:

```math
|\phi\rangle = \hat{U}|\psi\rangle
```

This works well if `\hat{U}` is an operator on the same number of qubits as `|\psi\rangle` represents. How would we apply a single qubit gate to a multi-qubit circuit? To make this more concrete, imagine we wanted to do the following

```math
\begin{align*}
|\psi\rangle &= \psi_0 |00\rangle + \psi_3 |11 \rangle \\
|\phi\rangle &= X_1 |\psi\rangle \\
\end{align*}
```
Here we want to apply an `X` gate only on qubit 1, while leaving qubit 0 untouched. In this case it's easy enough to do, as we remember that `X_1` is itself a bit of a shorthand for `I \otimes X` -- we just keep the identity hidden most of the time. So we can construct a 2-qubit matrix to apply `X_1` like so:

```math
|\phi\rangle = (I_0 \otimes X_1) |\psi\rangle
```

Let's write some code to do this.
"

# ╔═╡ c0e6e8ee-9ce6-4f5d-b5c6-e964eb9d8f50
md"
First, let's write down the matrix representation of some simple gates - for now we'll focus on the ones without angles/parameters.
"

# ╔═╡ 9a82d5ec-c36a-4a96-9c6e-70c344179801
const X = complex.([0 1; 1 0])

# ╔═╡ 2331d6d7-7c09-4fa4-b6e5-a9a5aa144a1a
const Y = [0 -im; im 0]

# ╔═╡ 462bbc01-26a1-4f9c-8101-4e35bdb29084
const Z = complex.([1 0; 0 -1])

# ╔═╡ 811f0d69-d1d7-44fb-9e83-9f1fc434fd95
const I = complex.([1 0; 0 1])

# ╔═╡ c21d83bf-fa20-424e-b9cf-3171c6162151
const H = complex.(1/√2 * [1 1; 1 -1])

# ╔═╡ ade1bd46-f929-47b5-a3f3-47104a64c177
function apply_gate_naive(ψ::Vector, gate::Matrix, gate_qubit::Int)
	n_qubits  = Int( log2(length(ψ)) )
	all_gates = [I for qubit in 1:n_qubits]
	all_gates[n_qubits - gate_qubit] = gate # Julia is one indexed!
	full_gate = reduce(kron, all_gates)
	return full_gate * ψ
end

# ╔═╡ e3f1bbbc-b77f-415e-821d-7049d000ff0a
md"
Now let's see how well this works. From using our brains, we can see that
```math
\begin{align*}
|\phi\rangle &= X_1 |\psi\rangle \\
&= X_1 (\psi_0 |00\rangle + \psi_3|11\rangle) \\
&= \psi_0 |01\rangle + \psi_3 |10\rangle \\
\end{align*}
```

Is that similar to what we observe from our function?
"

# ╔═╡ 895b29ab-fd4c-48f0-b34c-5faa7bbc4ecc
ψ = [1/√2, 0, 0, -1/√2]

# ╔═╡ c224a2f3-3960-4247-a605-3439d618ca24
md"
Is this correct? Let's check by hand...

```math
\begin{align*}
|\psi\rangle &= \psi_0 |00\rangle + \psi_3 |11 \rangle \\
|\phi\rangle &= X_1 |\psi\rangle \\
=& \frac{1}{\sqrt{2}} X_1 \left(|00\rangle - |11 \rangle\right) \\
=& \frac{1}{\sqrt{2}} \left(|01\rangle - |10\rangle\right) \\
\end{align*}
```

Which matches what we got from our function!
"

# ╔═╡ 2a2039e3-3a6b-49ac-8737-5e97b6fcb935
md"""
### Multi qubit gates

Let's look a little closer at some multiqubit gates, such as `SWAP`. As the name implies, this swaps the states between the target qubits.
```math
\begin{align*}
\mathrm{SWAP} |\psi_2\rangle &= \mathrm{SWAP}\left(\psi_0 |00\rangle + \psi_1 |01\rangle + \psi_2 |10|rangle + \psi_3|11\rangle\right) \\
&= \psi_0 |00 \rangle + \psi_1 |10\rangle + \psi_2 |01\rangle + \psi_3 \rangle \\
\end{align*}
```
We can write this in matrix form as:
```math
\mathrm{SWAP} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}
```

One interesting thing to note about this: can you write `SWAP` as a Kronecker product of two single qubit gates?
"""

# ╔═╡ a52b8837-cd46-4452-a6eb-3d922411ff9e
md"
#### Exercise

How would you modify `apply_gate_naive` to handle gates on multiple (adjacent) qubits?
"

# ╔═╡ ee03bbbe-cfb4-4d11-a544-8e0ba30aba11


# ╔═╡ 7a654e8c-7abe-4623-a1db-fa99eb5bdb6b
md"
### Computing expectation values

In order to compute correlation functions or other quantities of interest, we need to be able to compute expectation values of Hermitian operators (observables). In Dirac notation, we'd write this as:

```math
\langle O \rangle = \langle \psi | \hat{O} | \psi \rangle
```

This is an inner product. One way to implement this would be to first compute
```math
|\phi\rangle = \hat{O}|\psi\rangle
```
then
```math
\langle O \rangle = \langle \psi | \phi \rangle
```

Although, as it happens, Julia's `LinearAlgebra` standard library has a helpful `dot` function that can compute $\langle x | A | y \rangle$ in one invocation, *including* properly conjugating `x` if it is complex-valued. Again, similar to gates, for now we can construct an observable that spans all qubits using `kron`.
"

# ╔═╡ 21fd60c6-8b65-4803-a920-6da39a4b3cfd
function compute_expval_naive(O::Matrix, o_qubit::Int, ψ::Vector)
	n_qubits = Int( log2(length(ψ)) )
	all_observables = [I for qubit in 0:n_qubits - 1]
	all_observables[o_qubit + 1] = O
	full_O = reduce(kron, all_observables)
	return dot(ψ, full_O, ψ)
end

# ╔═╡ 87826bb8-e120-4615-ba82-0d1e6fbdf0b6
md"
Let's again check that we get the right answer here. Again thinking for a moment, we see that
```math
\begin{align*}
|\phi_2\rangle =& \frac{1}{\sqrt{2}} |01\rangle - \frac{1}{\sqrt{2}} |10\rangle \\
\langle Z_0 \rangle =& \langle \phi_2 | \hat{Z_0} | \phi_2 \rangle \\
=& \frac{1}{2} \left(\langle 01 | \hat{Z}_0 | 01 \rangle - \langle 10 | \hat{Z}_0 | 01 \rangle - \langle 01 | \hat{Z}_0 | 10 \rangle + \langle 10 | \hat{Z}_0 | 10 \rangle  \right) \\
=& \frac{1}{2} \left(1 - 0 - 0 - 1\right) \\
=& 0 \\
\end{align*}
```
and from our function, we get...
"

# ╔═╡ 43e463e6-a9ea-4528-b2b1-4c2a9327c593
md"
Again, looks right!

### Exercise

Why not test that this function is correct for `X` and `Y` as well?
"

# ╔═╡ d38909f3-7b4d-4a6f-8e1c-c939203d587a


# ╔═╡ da55b2ed-b560-4ce6-a79f-2e1735e2500c
md"
### Computing circuit unitaries

This approach of computing

```math
\hat{U} = \displaystyle\prod_{i=1}^{N_g} \hat{U}_i 
```

where $\hat{U}_i$ is the $i$-th gate in the circuit -- effectively computes the combined unitary of all the gates and applies it in one large matrix-vector multiplication. This is pretty simple to implement, as we saw, but there are many drawbacks too.

### Some obvious problems with this technique

- **Memory scaling** -- the size of the circuit unitary will be $2^N \times 2^N$, so as $N$ becomes large, it becomes extremely expensive to store the unitary
- **Wasted compute** -- most of our quantum gates span only a few qubits (usually, 3 or fewer), especially for superconducting systems. Taking repeated outer products with arguments that are mostly the identity matrix is slow and (probably) unnecesary
- **Non-contiguous gates are a pain** -- if the gate can't easily be decomposed, we have to do something complex for the outer product

However, this technique can still be competitive for very small circuits (as we'll see) and it is very useful for testing purposes.
"

# ╔═╡ 948fa3b6-be17-4f5d-a680-d4f4fecca8e8
md"
### Applying gates to a state vector (less naive approach)

Let's return to our 2 qubit state vector $|\psi_2\rangle$:

```math
|\psi_2\rangle = \psi_0 |00\rangle + \psi_1 |01\rangle + \psi_2 |10\rangle + \psi_3 |11\rangle
```

When looking at this 4 element vector, we note that we could also reshape it into a `2 x 2` matrix, such that each dimension corresponds to one qubit. Let's show this:

```math
| \psi_2 \rangle = \begin{bmatrix} \psi_0 & \psi_2 \\ \psi_1 & \psi_3 \end{bmatrix}
```

You can see, given our encoding above, that the rows of this matrix correspond to the two possible states of qubit 0, and the columns to qubit 1.

Now suppose we had 3 qubits, so a state vector with 8 elements.

```math
| \psi_3 \rangle = \psi_0 |000\rangle + \psi_1 |001\rangle + \psi_2 |010\rangle + \psi_3 |011\rangle + \psi_4 |100\rangle + \psi_5 |101\rangle + \psi_6 |110\rangle + \psi_7 |111\rangle
```

We can reshape this into a `2 x 4` or `4 x 2` object. This will allow us to apply gates without performing lots of unnecessary outer products.

Now we'll need to think a little bit carefully about how Julia arrays are laid out and how this corresponds to our qubit ordering. Julia arrays are laid out in [column-major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order). Let's take a look at how reshaping our state vector might affect things:
"

# ╔═╡ 9ea2a657-a6bd-44ea-ad88-133fd8906bae
# unnormalized - for demonstation purposes only!
ψ3 = [0.1, 0.1im, 0.2, -0.2im, 0.3, 0.3im, 0.4, -0.4im];

# ╔═╡ 99bc23ab-3327-4668-8252-eb4ff080c18b
for (ix, c) in enumerate(ψ3)
	println("Index: $ix, $c")
end

# ╔═╡ cdc44a67-b573-4fb2-a68f-1c7b5d62958e
ϕ3 = reshape(ψ3, (2, 2, 2))

# ╔═╡ 45d1b011-c2ed-4c8f-aa29-cba8bf24bb67
φ3 = reshape(ψ3, (4, 2))

# ╔═╡ 29da18fe-0935-4cb0-8f99-9791bce298a5
ω3 = reshape(ψ3, (2, 4))

# ╔═╡ 25bbcd81-43e4-4de8-a6bf-a2d3456e72c1
md"
Let's now examine how we can use this to apply gates more efficiently. Consider again applying `X` to various qubits. For safety, we'll work out by hand what the answer ought to be:

```math
\begin{align*}
\hat{X}_0|\psi\rangle =& \hat{X}_0 \left(0.1 |000\rangle + 0.1\imath |001\rangle + 0.2 |010\rangle - 0.2\imath |011\rangle + 0.3 |100\rangle + 0.3\imath |101\rangle + 0.4 |110\rangle - 0.4\imath |111\rangle\right) \\
=& 0.1 |001\rangle + 0.1\imath |000\rangle + 0.2 |011\rangle - 0.2\imath |010\rangle + 0.3 |101\rangle + 0.3\imath |100\rangle + 0.4 |111\rangle - 0.4\imath |110\rangle \\
=& 0.1\imath |000\rangle + 0.1 |001\rangle - 0.2\imath |010\rangle + 0.2 |011\rangle + 0.3\imath |100\rangle + 0.3 |101\rangle - 0.4\imath |110\rangle + 0.4 |111\rangle\ \\
\hat{X}_1|\psi\rangle =& 0.1 |010\rangle + 0.1\imath |011\rangle + 0.2 |000\rangle - 0.2\imath |001\rangle + 0.3 |110\rangle + 0.3\imath |111\rangle + 0.4 |100\rangle - 0.4\imath |101\rangle \\
=& 0.2 |000\rangle - 0.2\imath |001\rangle + 0.1 |010\rangle + 0.1\imath |011\rangle + 0.4 |100\rangle - 0.4\imath |101\rangle + 0.3 |110\rangle + 0.3\imath |111\rangle \\
\hat{X}_2|\psi\rangle =& 0.1 |100\rangle + 0.1\imath |101\rangle + 0.2 |110\rangle - 0.2\imath |111\rangle + 0.3 |000\rangle + 0.3\imath |001\rangle + 0.4 |010\rangle - 0.4\imath |011\rangle \\
=& 0.3 |000\rangle + 0.3\imath |001\rangle + 0.4 |010\rangle - 0.4\imath |011\rangle + 0.1 |100\rangle + 0.1\imath |101\rangle + 0.2 |110\rangle - 0.2\imath |111\rangle    \\
\end{align*}
```


In this new approach, $ψ3$ is reshaped such that the 2nd dimension represents the states on qubit 0. Here's how we can check:
"

# ╔═╡ 9165bec9-c0ee-4606-aafa-ffe1f6eb32b5
function apply_gate_reshaped1(ψ::Vector, gate::Matrix, gate_qubit::Int)
	ψ_reshaped = reshape(ψ, (4, 2))
	return vec(ψ_reshaped * gate)
end

# ╔═╡ 113037cb-6431-431d-af3a-840e5fd47473
ϕ_reshaped1 = apply_gate_reshaped1(ψ3, X, 0)

# ╔═╡ 2eaf59b7-fa3b-47f4-b662-fbf4183e0ee5
md"
Great! Looks like they match. But what if we apply `X` to a different qubit, say qubit 1?
"

# ╔═╡ 2f50fb28-a328-4a30-b048-8b901738624c
ϕ_reshaped2 = apply_gate_reshaped1(ψ3, X, 1)

# ╔═╡ 0cd36d05-41fb-441a-9307-6367e53a721e
md"
Uh oh. This is wrong! Why?

The problem is that we reshaped without *permuting*. As said above, if we simply reshape as we did in `apply_gate_reshaped1`, we're implicitly applying the gate to qubit 0 every time. We need to **permute** the dimensions of the input state vector $|\psi\rangle$ (and be careful to permute back!) in order to handle gates appropriately.
"

# ╔═╡ 93a3be9d-ddef-473a-8997-78091b176275
function apply_gate_reshaped2(ψ::Vector, gate::Matrix, gate_qubit::Int)
	n_qubits    = Int( log2(length(ψ)) )
	ψ_reshaped  = reshape(ψ, ntuple(i->2, n_qubits))
	permutation = collect(1:n_qubits)
	permutation[end] = gate_qubit + 1
	permutation[gate_qubit + 1] = n_qubits
	ϕ           = permutedims(ψ_reshaped, permutation)
	ϕ_reshaped  = reshape(ϕ, (2^(n_qubits - 1), 2))
	ϕ_reshaped *= gate
	ϕ           = reshape(ϕ_reshaped, ntuple(i->2, n_qubits))
	ψ_reshaped  = permutedims(ϕ, permutation)	
	ϕ           = reshape(ψ_reshaped, 2^n_qubits)
	return ϕ
end

# ╔═╡ 06052b8c-03a7-47c8-b8cd-a122db0e0bea
ϕ_reshaped3 = apply_gate_reshaped2(ψ3, X, 1)

# ╔═╡ 18d097b3-a9ed-4910-a201-e48d5a836f6b
md"
That looks better! Just for safety, let's check a gate applied to qubit 2 as well.
"

# ╔═╡ 510bae1b-cd56-4844-aab7-cbf1a23b31c2
ϕ_reshaped4 = apply_gate_reshaped2(ψ3, X, 2)

# ╔═╡ d9a62485-330e-4367-999e-44ea2d75397e
md"
This matches up with what we calculated above, and our code is consistent with itself. Huzzah!
"

# ╔═╡ e13154d5-e3a9-46eb-836d-8cbe15630002
md"
### Exercise

How would you modify `apply_gate_reshaped2` to handle gates on multiple qubits, including **non-contiguous** qubits?
"

# ╔═╡ 60ec9535-59d1-4545-af77-4b21d0bb811a


# ╔═╡ 18b15e49-b733-4104-a031-42b7b8a53acc
md"
### Performance on the brain

Above, I claimed that our \"naive\" approach has lots of performance problems. Is this reshape-permute-reshape approach any better? Rather than speculating, let's find out.
"

# ╔═╡ c87c7cd5-7234-4006-8a54-2510a59dbeb7
md"
We're going to create some convenience functions here in order to measure performance of our function specifically, and not any setup or teardown work.
"

# ╔═╡ 9e9d8943-57ef-47fa-90fd-81d69f346bd8
ψ_small() = normalize!([0.1, 0.1im, 0.2, -0.2im, 0.3, 0.3im, 0.4, -0.4im])

# ╔═╡ e67e0a89-d3ab-431a-a640-9c9bf40acf3d
reshaped_runner(ψ) = apply_gate_reshaped2(ψ, X, 2)

# ╔═╡ c2c830fa-f6f6-4b70-b487-4913f93bdd04
@b ψ_small() reshaped_runner

# ╔═╡ fd2f0aea-06cc-41a2-a781-6f3b4b126e11
md"
Reshaping isn't doing so well for a small (3-qubit) circuit. What about something larger?
"

# ╔═╡ 186daf87-d823-4907-93ee-ad82ec667f70
ψ_big() = normalize!(rand(ComplexF64, 2^12));

# ╔═╡ 6da0833a-c1ef-46d5-826e-f8b26d0da783
@b ψ_big() reshaped_runner

# ╔═╡ 9e8b447e-7e3e-4d99-890e-9140388153ee
md"
Now we do see a difference... partly because of allocations! But not only for that reason - matrix-vector multiplication scales as (roughly) $\mathcal{O}\left(n^3\right)$, so avoiding constructing massive matrices and multiplying them is in our interest.

Another important lesson here: **always** benchmark (or, rather, chairmark!). Our intuitions about physics and computation are often wrong -- as three luminaries once said,

> Premature optimization is the root of all evil.

Donald Knuth

> The great thing about physical intuition is it can be adjusted to fit the facts.

Roger Penrose

> More is different.

P. W. Anderson

"

# ╔═╡ 2510c7f7-55b8-49e8-bc0e-ce59952a4079
md"
### Exercise

Can you adjust the `compute_expval_naive` function above to use this `reshape`-`permute`-`reshape` scheme?
"

# ╔═╡ af671a1c-dd98-4046-a063-5e6a42e1174f


# ╔═╡ 3bbc13c8-ff21-4404-8600-f3922f65c961
md"
## Becoming less naive over time

Although we've achieved *better* performance with our new approach, we can in fact do even more. The trick lies in realizing that:
- `permutedims` requires a copy
- If we are clever and a bit careful about using indices, we can avoid this copy
- As a bonus, we can use (and control) multithreading

How is this possible? By exploiting the fact that *integers* on classical computesr are also represented as *bitstrings* \"under the hood\". We've already set the stage for this with our coefficient labelling - $\psi_0$, $\psi_3$, and the like. Let's dig a little deeper.

Julia has several inbuilt functions which we can use to inspect how integers are represented internally. We'll focus on [`digits`](https://docs.julialang.org/en/v1/base/numbers/#Base.digits) and [`bitstring`](https://docs.julialang.org/en/v1/base/numbers/#Base.bitstring). First, we'll examine indices `0` through `7`, for our 3-qubit state vector. (Why do you think we start **these** indices at 0?)
"

# ╔═╡ 77f4ca16-7fed-4025-b9b8-a00c06eb1d5f
for ix in 0:7
	println("Digits of index $ix: $(digits(ix, pad=3, base=2))")
end

# ╔═╡ 71bc7a01-7971-4443-9ee9-db2dd289bcaa
for ix in 0:7
	println("Full bitstring of index $ix: $(bitstring(ix))")
end

# ╔═╡ 3e148898-4ec3-424b-8885-7689109512ba
md"
`bitstring` shows the **full** representation of an integer in memory. What do you notice about this representation, considering our previous discussion about [endianness](https://en.wikipedia.org/wiki/Endianness)?

#### Brief digression for physicists (CS majors do not interact)
All integers, floats, indeed all numbers are represented in computer memory as *bytes*, which are groups of 8 *bits*. Thus, an `Int64` in Julia is composed of **eight** bytes, and an `Int32` of **four** bytes.


Here we need to introduce one more concept: **bit-shifting**.

In Julia, as in many other languages, we have two bit shift operators: [`>>`](https://docs.julialang.org/en/v1/base/math/#Base.:%3E%3E) and [`<<`](https://docs.julialang.org/en/v1/base/math/#Base.:%3C%3C). What do each of these do?
"

# ╔═╡ d7bfaa44-f577-4487-8de6-77b137871d9c
for ix in 0:7
	println("Full bitstring of index $ix: $(bitstring(ix))")
	println("Full bitstring of index $ix shifted left $(bitstring(ix << 1))")
	println("Full bitstring of index $ix shifted right $(bitstring(ix >> 1))")
	println()
end

# ╔═╡ 8053adfd-cddf-4b3d-9a0f-fcb2bd23e42c
md"
That's very nice, but how can we use this to compute gate applications or expectation values more efficiently?

Consider again the application of a generic single qubit gate $G$ on qubit 0. The matrix representation of $G$ looks like:

```math
G = \begin{bmatrix} g_{00} & g_{01} \\ g_{10} & g_{11} \end{bmatrix}
```

So, how can we apply this in an allocation-free way? Let's take a look at our 3-qubit example above. We saw that by reshaping an `(8,1)` vector into a `(4,2)` matrix, we could target qubit zero. In fact, we can perform the same operation **without reshaping** using bitshifting, and in a more general way.

This is somewhat subtle, so let's go through it step by step.

For a `2^N`-length statevector, and a single qubit gate, we need to \"connect\" states $|0\rangle$ and $|1\rangle$ on that qubit -- meaning there are $2^{N-1}$ \"connections\" to make (as we saw in the reshaping above). First, we can examine:

```julia
for ix in 0:(2^n_qubits)-1
	...
end
```

As we did above. But we see, by considering the bitstrings, that we'll \"double target\" the qubit of interest, because we pick up both the bitstrings that contain a `0` and a `1` on that qubit. Instead, we can consider 

```julia
for ix in 0:2^(n_qubits-1)-1
	...
end
```

But how to target the qubit we do want? Let's say we want to apply a gate to qubit 1 on a 3-qubit statevector - if you print the `bitstring`s of all integers `0:4`, you'll see that qubits 0 and 1 would be touched - but not qubit 2. We can \"insert\" a zero bit **at the appropriate qubit**.

```julia
function expand_int(ix::Ti, qubit) where {Ti}
    left  = (ix >> qqubit) << qubit
    right = ix - left
    return (left << one(Ti)) ⊻ right
end

for ix in 0:2^(n_qubits-1)-1
	expand_int(ix, 1)
end
```

What? What does this actually do - let's take a closer look:
"

# ╔═╡ b7532daa-a526-421a-8db9-4c4539c0c969
function expand_int(ix::Ti, qubit) where {Ti}
    left  = (ix >> qubit) << qubit
    right = ix - left
    return (left << one(Ti)) ⊻ right
end

# ╔═╡ 2d9f1ca6-0855-4104-9d5a-c8bd51880dff
for ix in 0:4
	println("bitstring(ix): $(bitstring(ix))), bitstring expanded ix: $(bitstring(expand_int(ix, 1)))")
end

# ╔═╡ afce6fe8-7876-4766-899c-a3bc8a35451e
md"
Okay ... so we bumped the bits to the left starting at index 1? Reckoning from the left? If you're still confused, it can be worth playing with a variety of integers and \"expansion qubits\" to get a more intuitive feel for what this is doing.

Now we also need a way to \"connect\" to the 1-valued indices. This can be achieved by \"flipping\" the qubit we want to act on, from 0 to 1. This is simpler than the expansion operation:

```julia
flip_bit(ix::Ti, q) where {Ti} = ix ⊻ (one(Ti) << q)
```

This `⊻` is `XOR`, or \"exclusive or\" (one or the other, but not both), such that:
- `0 ⊻ 0 = 0`
- `0 ⊻ 1 = 1`
- `1 ⊻ 0 = 1`
- `1 ⊻ 1 = 0`

The shifting of `one(Ti)` is to flip the appropriate bit - again, try looping through a few integers to see how this works.

With this, we have the tools we need to write an efficient method for applying arbitrary single qubit gates.
"

# ╔═╡ 0fb793b9-b3cc-4d97-b6e6-5407ab3c23c2
flip_bit(ix::Ti, q) where {Ti} = ix ⊻ (one(Ti) << q)

# ╔═╡ afafd52e-72b4-4a7b-8cc5-f7fbbf218c39
function apply_gate_shifting1(ϕ::Vector, ψ::Vector, gate::Matrix, gate_qubit::Int)
	n_qubits = Int( log2(length(ψ)) )
	for ix in 0:2^(n_qubits-1)-1
		amplitude_0 = expand_int(ix, gate_qubit)
		amplitude_1 = flip_bit(amplitude_0, gate_qubit)
		# Julia is one-indexed!
		amplitude_0 += 1
		amplitude_1 += 1
		old_ψ_0 = ψ[amplitude_0]
		old_ψ_1 = ψ[amplitude_1]
		ϕ[amplitude_0] = gate[1, 1] * old_ψ_0 + gate[1, 2] * old_ψ_1
		ϕ[amplitude_1] = gate[2, 1] * old_ψ_0 + gate[2, 2] * old_ψ_1
	end
	return ϕ
end

# ╔═╡ 7ea1ed14-dab6-47ae-970c-7dee8b62ca1a
md"
Let's check again that this returns correct results.
"

# ╔═╡ 49d94720-e428-41c8-a5a6-035a4472c555
apply_gate_shifting1(copy(ψ3), ψ3, X, 0)

# ╔═╡ ae670d56-2979-40a6-b71a-b49c15ef6613
apply_gate_shifting1(copy(ψ3), ψ3, X, 1)

# ╔═╡ 407f0764-1388-454a-b132-8d28130a18c4
apply_gate_shifting1(copy(ψ3), ψ3, X, 2)

# ╔═╡ fb2f9abe-0f6f-4c58-a46e-90e339d8ac98
md"
Again, looks correct! In our example here, we created an unnecessary copy of $|ψ\rangle$ for testing convenience. In practice, we would simply re-use the vector -- our approach of stepping over `2^(n_qubits - n_gate_qubits)` amplitudes ensures we don't \"double touch\" any indices. Let's now compare the performance using a copy-free approach.
"

# ╔═╡ 6f8f5a92-2421-4f89-b165-010db2ef6f98
function apply_gate_shifting(ψ::Vector, gate::Matrix, gate_qubit::Int)
	n_qubits = Int( log2(length(ψ)) )
	for ix in 0:2^(n_qubits-1)-1
		amplitude_0 = expand_int(ix, gate_qubit)
		amplitude_1 = flip_bit(amplitude_0, gate_qubit)
		# Julia is one-indexed!
		amplitude_0 += 1
		amplitude_1 += 1
		old_ψ_0 = ψ[amplitude_0]
		old_ψ_1 = ψ[amplitude_1]
		ψ[amplitude_0] = gate[1, 1] * old_ψ_0 + gate[1, 2] * old_ψ_1
		ψ[amplitude_1] = gate[2, 1] * old_ψ_0 + gate[2, 2] * old_ψ_1
	end
	return ψ
end

# ╔═╡ a7df0f04-d43d-42e7-afdf-6da9984a30bc
@b ψ_small() reshaped_runner

# ╔═╡ aebc17c3-f762-477e-a26a-519c701ffcb9
shifting_runner(ψ) = apply_gate_shifting(ψ, X, 2)

# ╔═╡ 4f78fb80-6b17-4f5f-b47d-e5d55aaa7764
@b ψ_small() shifting_runner

# ╔═╡ 2b5428cc-8890-42b3-91e9-d3b5801ac2d8
md"
Nice! Even for a very small state vector, this approach works nicely -- can you suggest a reason why? Let's now examine our larger statevector.
"

# ╔═╡ 4008b027-96f5-4b2f-beeb-de4a5fa524c9
@b ψ_big() reshaped_runner

# ╔═╡ 299ddda6-b7b1-4d72-9a2c-4aac5eefaf8c
@b ψ_big() shifting_runner

# ╔═╡ e8fb90e4-f741-4125-9e67-1ea9d340c018
md"
Another advantage of our approach is that we can use multithreading. In fact, the `LinearAlgebra` routines we are calling in `apply_gate_naive` and `apply_gate_reshaped2` already use BLAS behind the scenes, which does have support for multi-threading.

Is writing our own threading logic for `apply_gate_shifting` worthwhile? Let's try it and benchmark to determine the answer.
"

# ╔═╡ 12c26fb0-0c83-4beb-a367-3bbfa635ae61
function apply_gate_shifting_threaded(ψ::Vector, gate::Matrix, gate_qubit::Int)
	n_qubits = Int( log2(length(ψ)) )
	Threads.@threads for ix in 0:2^(n_qubits-1)-1
		amplitude_0 = expand_int(ix, gate_qubit)
		amplitude_1 = flip_bit(amplitude_0, gate_qubit)
		# Julia is one-indexed!
		amplitude_0 += 1
		amplitude_1 += 1
		old_ψ_0 = ψ[amplitude_0]
		old_ψ_1 = ψ[amplitude_1]
		ψ[amplitude_0] = gate[1, 1] * old_ψ_0 + gate[1, 2] * old_ψ_1
		ψ[amplitude_1] = gate[2, 1] * old_ψ_0 + gate[2, 2] * old_ψ_1
	end
	return ψ
end

# ╔═╡ 40c345ea-6b3e-4fa1-a60c-b6abb2ef4e5a
md"
How many threads do we have available? Julia allows us to check. The answer will of course vary depending on your computer.
"

# ╔═╡ 0c261603-48ee-4b5a-8727-02d3623203f5
Threads.nthreads()

# ╔═╡ 6fe606a8-d91a-4b00-bd1c-a352623c3985
threaded_runner(ψ) = apply_gate_shifting_threaded(ψ, X, 2)

# ╔═╡ 0efb844f-7575-426d-b5f5-ee235b7f16e4
@b ψ_small() threaded_runner

# ╔═╡ 77bd3384-c224-45ba-acab-32dd01cedab1
@b ψ_big() threaded_runner

# ╔═╡ a016e2cb-1b5c-46c9-bfae-f35df2a294d2
md"
We do see a performance benefit, but not as much as you might expect from the raw thread count. Why is that?

Julia by default uses [dynamic threading](https://docs.julialang.org/en/v1/manual/multi-threading/), which means each instance of work (so, each `ix` in our loop above) is a Julia `Task`, which can be migrated among the threads processing the work. This can be very helpful -- it means that an idle thread can \"steal\" work from another thread with a large queue. Even better for our purposes, it means that we can **nest** threaded loops, so that a thread can itself spawn more threaded work to put on the scheduling queue. This is very helpful if we want to simulate multiple state vectors at once. The creation of these `Task`s is responsible for the new allocations. Additionally, if we spawn too many tasks, the Julia scheduler can become overwhelmed and performance suffers, though many improvements to the scheduler have been made recently which have mitigated this issue. So how many is \"too many\"? Again, the answer usually has to come from experimentation and benchmarking, and can change from system to system. If we do want to constrain the number of tasks, can we do so? Julia makes this relatively simple as well, using [`Iterators.partition`](https://docs.julialang.org/en/v1/base/iterators/#Base.Iterators.partition).
"

# ╔═╡ 700ef4ad-6053-45c0-a8ff-b8494d8be576
collect(Iterators.partition(0:15, 2))

# ╔═╡ 0a239d26-796d-4721-b2f9-c77b8c64b354
const chunk_size = 2^8

# ╔═╡ bab81d91-c3c1-44a0-8dee-2022b5df785d
function apply_gate_shifting_chunked(ψ::Vector, gate::Matrix, gate_qubit::Int)
	n_qubits  = Int( log2(length(ψ)) )
	chunk_ixs = collect(Iterators.partition(0:2^(n_qubits-1)-1, chunk_size))
	Threads.@threads for chunk_ix in 1:length(chunk_ixs)
		for ix in chunk_ixs[chunk_ix]
			amplitude_0 = expand_int(ix, gate_qubit)
			amplitude_1 = flip_bit(amplitude_0, gate_qubit)
			# Julia is one-indexed!
			amplitude_0 += 1
			amplitude_1 += 1
			old_ψ_0 = ψ[amplitude_0]
			old_ψ_1 = ψ[amplitude_1]
			ψ[amplitude_0] = gate[1, 1] * old_ψ_0 + gate[1, 2] * old_ψ_1
			ψ[amplitude_1] = gate[2, 1] * old_ψ_0 + gate[2, 2] * old_ψ_1
		end
	end
	return ψ
end

# ╔═╡ fabf4d37-ecaa-40e4-bc18-23c928b48cbe
chunked_runner(ψ) = apply_gate_shifting_chunked(ψ, X, 2)

# ╔═╡ c3efeba8-722e-4962-b072-e2d8212e40a0
@b ψ_small() chunked_runner

# ╔═╡ 189f5812-a095-4371-b6e8-fe5bf5a9af5c
@b ψ_big() chunked_runner

# ╔═╡ c4cb5ee6-ae5f-4918-a485-4e7ae8ccfc8c
md"
This didn't seem to help much. What if we try an even bigger state vector?
"

# ╔═╡ 5afc29e6-c4ec-4c3d-b080-051eb7caa1db
ψ_very_big() = normalize!(rand(ComplexF64, 2^20));

# ╔═╡ f99154d6-0b48-484c-a36c-2f2b4e099e1c
@b ψ_very_big() threaded_runner

# ╔═╡ 7c2ede52-bcc6-4c1c-bc56-ebc9ca37b126
@b ψ_very_big() chunked_runner

# ╔═╡ 5f0eb332-d979-4df1-b57f-24e1e2b6680b
md"
Still not so dramatic! But if we try the nested threading situation, what do we observe?
"

# ╔═╡ 040f885a-8ddb-4254-978a-bc7420ec7877
several_ψ_bigs() = [normalize!(rand(ComplexF64, 2^12)) for i in 1:100];

# ╔═╡ eb099f55-a8b4-48f3-87cb-644330b150e9
@b several_ψ_bigs() (ψ->begin
	Threads.@threads for ψ_ix in 1:100
		apply_gate_shifting_threaded(ψ[ψ_ix], X, 2)
	end
end)

# ╔═╡ d18e277f-ecf6-4d4f-a6d4-f8404d0df59b
@b several_ψ_bigs() (ψ->begin
	Threads.@threads for ψ_ix in 1:100
		apply_gate_shifting_chunked(ψ[ψ_ix], X, 2)
	end
end)

# ╔═╡ bad3f02f-9685-476b-9ae3-7c455b91b993
md"
Again we don't see much benefit! This shows first, the power of the Julia scheduler, and second, the use of actually benchmarking to confirm our ideas. Are there other optimizations we could try? One easy target is [bounds checking](https://docs.julialang.org/en/v1/devdocs/boundscheck/). These checks ensure that all array access is \"inbounds\", but if we're certain that our accesses are, we can turn off the bounds checks and hopefully see some benefit.
"

# ╔═╡ 6a2f6927-8a01-40d9-a512-0f75d4f2520a
function apply_gate_shifting_inbounds(ψ::Vector, gate::Matrix, gate_qubit::Int)
	n_qubits = Int( log2(length(ψ)) )
	Threads.@threads for ix in 0:2^(n_qubits-1)-1
		amplitude_0 = expand_int(ix, gate_qubit)
		amplitude_1 = flip_bit(amplitude_0, gate_qubit)
		# Julia is one-indexed!
		amplitude_0 += 1
		amplitude_1 += 1
		@inbounds begin
			old_ψ_0 = ψ[amplitude_0]
			old_ψ_1 = ψ[amplitude_1]
			ψ[amplitude_0] = gate[1, 1] * old_ψ_0 + gate[1, 2] * old_ψ_1
			ψ[amplitude_1] = gate[2, 1] * old_ψ_0 + gate[2, 2] * old_ψ_1
		end
	end
	return ψ
end

# ╔═╡ a2d44668-bbbd-4963-b283-0b0929e2bf00
inbounds_runner(ψ) = apply_gate_shifting_inbounds(ψ, X, 2)

# ╔═╡ dbf2ef8b-afce-4adb-9999-312a6a5eb7a0
@b ψ_small() inbounds_runner

# ╔═╡ af0e7a78-231c-48e7-b116-356ef92dee28
@b ψ_big() inbounds_runner

# ╔═╡ d8ce2c16-0556-4c7d-bafe-a6efaa46a1a0
@b ψ_very_big() inbounds_runner

# ╔═╡ 5790bb0f-1f6a-4278-bce5-87fc0bb33a92
md"
It seems none of these techniques did too much to help. In that case, we can try profiling to understand where the code is spending its time.
"

# ╔═╡ 58595934-5389-434a-9f33-538d4b71c31e
begin
	Profile.clear()
	φ = ψ_very_big()
	@profile inbounds_runner(φ)
	Profile.print()
end

# ╔═╡ 47ad6221-e5bd-4f77-b8ad-28231d30b315
md"
Looking at this call graph, we can see a few things:
  - We're spending a **bunch** of time on float promotion (probably from the element types of `X`)
  - There's also some time being spent on the `expand_int` function, which might be good to look at once we fix the float promotion issues
"

# ╔═╡ 13a83ffa-666f-42f6-9f62-a8088f512bc9
function apply_gate_shifting_nopromo(ψ::Vector{ComplexF64}, gate::Matrix{ComplexF64}, gate_qubit::Int)
	n_qubits = Int( log2(length(ψ)) )
	Threads.@threads for ix in 0:2^(n_qubits-1)-1
		amplitude_0 = expand_int(ix, gate_qubit)
		amplitude_1 = flip_bit(amplitude_0, gate_qubit)
		# Julia is one-indexed!
		amplitude_0 += 1
		amplitude_1 += 1
		@inbounds begin
			old_ψ_0 = ψ[amplitude_0]
			old_ψ_1 = ψ[amplitude_1]
			ψ[amplitude_0] = gate[1, 1] * old_ψ_0 + gate[1, 2] * old_ψ_1
			ψ[amplitude_1] = gate[2, 1] * old_ψ_0 + gate[2, 2] * old_ψ_1
		end
	end
	return ψ
end

# ╔═╡ 456976ea-7e5f-4cc0-ad5e-1bfef8b1da01
md"If we try to call this with our `X` defined above, we'll get a `MethodError` because the element types of `X` are `Complex{Int}`. Let's define a new matrix which has floating point elements."

# ╔═╡ 479df144-24f9-4966-b2a6-39c377ee4af8
const X_float = complex.([0.0 1.0; 1.0 0.0])

# ╔═╡ b11f1f4c-04a9-43b9-af1c-3e5bc560dd06
nopromo_runner(ψ) = apply_gate_shifting_nopromo(ψ, X_float, 2)

# ╔═╡ 242a2d57-2478-4177-8036-a3683ecb947d
@b ψ_very_big() inbounds_runner

# ╔═╡ e4691b67-1d63-49df-960e-954806c49970
@b ψ_very_big() nopromo_runner

# ╔═╡ 4a4d7a98-e1b0-432f-acf5-00a76a05eb6d
md"Well, that does look better! An interesting lesson here is that we should have focused on the type instability issues first (which we can detect through profiling) and then gone for the more marginal gains outlined above. This is why profiling and benchmarking are very helpful, and also illustrates how a convenient choice early on can hamper performance later. Let's profile this new function to see where it's now spending time."

# ╔═╡ cf13f425-c34c-4d6f-9605-e57f677414c9
begin
	Profile.clear()
	ω = ψ_very_big()
	@profile nopromo_runner(ω)
	Profile.print()
end

# ╔═╡ faf6d35f-55de-477b-b7c0-d01497776f43
md"
Here we can see that a lot of the time is spent on unavoidable work - the arithmetic operations to compute the new elements of $|\psi\rangle$. We are, however, still spending some time in `getindex` that we might be able to improve upon. When we're indexing the matrix `g`, Julia is performing some work to figure out the linear index in memory corresponding to our Cartesian two-part index. We can help Julia save some steps here by precomputing this, keeping in mind that Julia arrays are column-major.
"

# ╔═╡ 897b47c1-6bd7-4fdb-ac53-2506a0a9d959
function apply_gate_shifting_linear(ψ::Vector{ComplexF64}, gate::Matrix{ComplexF64}, gate_qubit::Int)
	n_qubits = Int( log2(length(ψ)) )
	Threads.@threads for ix in 0:2^(n_qubits-1)-1
		amplitude_0 = expand_int(ix, gate_qubit)
		amplitude_1 = flip_bit(amplitude_0, gate_qubit)
		# Julia is one-indexed!
		amplitude_0 += 1
		amplitude_1 += 1
		@inbounds begin
			old_ψ_0 = ψ[amplitude_0]
			old_ψ_1 = ψ[amplitude_1]
			ψ[amplitude_0] = gate[1] * old_ψ_0 + gate[3] * old_ψ_1
			ψ[amplitude_1] = gate[2] * old_ψ_0 + gate[4] * old_ψ_1
		end
	end
	return ψ
end

# ╔═╡ 91cfbca9-d11d-43b7-bd9a-892a66f0db21
linear_runner(ψ) = apply_gate_shifting_linear(ψ, X_float, 2)

# ╔═╡ 84d88d31-fa73-4650-bb31-8f0d040e5ece
begin
	Profile.clear()
	ξ = ψ_very_big()
	@profile linear_runner(ξ)
	Profile.print()
end

# ╔═╡ c59dde08-79c3-472e-bd6d-5683b38521fe
md"
Now we really do seem to be spending the bulk of our time in \"useful work\", generating indices and new amplitudes from old ones.
"

# ╔═╡ 2a70ea35-6302-4748-85f5-4616899d5082
md"
## Multi-qubit gates

Now we're ready to venture forth into the wild, exciting world of gates on more than one qubit. Examples of such gates are:
  - `SWAP`
  - `XX`
  - `YY`
  - `XY`
  - `ZZ`
  - `CPhaseShift` and friends

In this section, you're going to have less step-by-step guidance, but hopefully you can extend what we've already done for a single qubit. Again, it will probably be helpful to write an `apply_gate_naive` which can handle contiguous qubits as a correctness checker.

One thing to consider, when using `expand_int`, is the order of the qubits. Let's look at a small test case:
"

# ╔═╡ 2597cc50-8fc7-4363-9fdd-da2e27bc3d09
begin
	a = Int8(5)
	println("Initial bitstring: $(bitstring(a))")
	# let's expand this at qubits 1 and 3
	println("Expanded in order (1, 3) bitstring: $(bitstring(expand_int(expand_int(a, 1), 3)))")
	println("Expanded in order (3, 1) bitstring: $(bitstring(expand_int(expand_int(a, 3), 1)))")
end

# ╔═╡ 2c4202aa-230b-44e5-a323-a62788a55212
md"
The results are **not** the same! Thus, we'll need to be careful of what order we insert bits. Let's look step by step to see what's happening here:
"

# ╔═╡ 02645015-f490-4251-935d-0c076c5b40d5
bitstring(expand_int(Int8(5), 1))

# ╔═╡ c2a87355-a575-4732-8dee-3a9b6ca58eed
md"By inserting at position 1, we effectively shift all the digits to the left of 1 over. This means the digit that was formerly at position 3 is now at position 4."

# ╔═╡ 8d604aa0-a871-424e-a702-1c4dde4ee978
md"
### Writing a two qubit gate kernel - tips

- For a single qubit function, we looped over the total number of amplitudes divided by 2. For a two qubit function, what would be an appropriate number of indices?
- Be careful of the order in which you flip bits to generate indices for the innermost matrix-vector multiplication/update of `ψ`
- We can reuse the name `apply_gate_shifting_linear` if we want, due to Julia's [multiple dispatch](https://docs.julialang.org/en/v1/manual/methods/#Methods) feature. That will allow us to pass arbitrary gates and qubits and have Julia pick the most appropriate inner method for us. Another option is to define a simple `apply_gate` function which then calls `apply_gate_shifting_linear` fot the single qubit case, or rename that function.
- It may be worth adding a few checks to ensure the size of the input gate matrix fits the number of qubits, that the qubits are not identical, etc. and throwing errors if you encounter these.

"

# ╔═╡ 4e36b069-5d4b-416c-9b5d-baad75e40f01
function apply_gate(ψ::Vector{ComplexF64}, gate_matrix::Matrix{ComplexF64}, qubit1::Int, qubit2::Int)
	# some stuff goes here!
	return ψ
end

# ╔═╡ f26c5049-d7be-49f1-9f9e-5e9f91230b0f
md"
### Handling controlled gates

What we've developed for a single qubit gate can be generalized to controlled gates or gates on arbitrary numbers of qubits, now that we have a sense of what to watch for.

Controlled gates, like `CNOT` (a.k.a `CX`), apply a gate to the target qubit(s) if and only if the control qubit(s) are in a certain state ($|1\rangle for \"control\", $|0\rangle for \"negative control\"). In matrix form, we can represent `CNOT` as:

```math
\mathtt{CNOT} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}
```

We could certainly apply this as a two-qubit gate, but there's also a way to extend our single qubit function above and avoid many unnecessary arithmetic operations. In the controlled gate case, only states with the $|1\rangle$ state on the control qubit have their amplitudes changed, so if we target those states only we can cut the number of operations in half. We can do this by using our `expand_int` and `flip_bit` functions.
"

# ╔═╡ 44e4179a-d141-412c-8791-9e35b268f6c2
function apply_controlled_gate(ψ::Vector{ComplexF64}, gate::Matrix{ComplexF64}, gate_qubit::Int, control_qubit::Int)
	n_qubits = Int( log2(length(ψ)) )
	small_q, big_q = minmax(gate_qubit, target_qubit)
	# why is this now n_qubits - 2 ?
	Threads.@threads for ix in 0:2^(n_qubits-2)-1
		amplitude_0 = expand_int(ix, small_q)
		amplitude_0 = expand_int(amplitude_0, big_q)
		amplitude_0 = flip_bit(amplitude_0, control_qubit)
		amplitude_1 = flip_bit(amplitude_0, gate_qubit)
		amplitude_0 += 1
		amplitude_1 += 1
		@inbounds begin
			old_ψ_0 = ψ[amplitude_0]
			old_ψ_1 = ψ[amplitude_1]
			ψ[amplitude_0] = gate[1] * old_ψ_0 + gate[3] * old_ψ_1
			ψ[amplitude_1] = gate[2] * old_ψ_0 + gate[4] * old_ψ_1
		end
	end
	return ψ
end

# ╔═╡ 02145b62-1bd2-4f74-873b-48bce6a76aa2
md"
## Exercise

Can you extend this `apply_controlled_gate` function to handle negatively controlled gates? How about gates with arbitrary numbers of control qubits? What about controlled multi-qubit gates?
"

# ╔═╡ 8050906c-95fc-47dc-a755-5d1ec354fd41


# ╔═╡ f692016d-ae34-4c86-b5ab-e8d718c6603c
md"
## Expectation values using bit-shifting

Similar to what we've done above for gate application, we can write an efficient method for computing expectation values of observables using mostly the same logic. As a reminder, we need to compute:

```math
\langle O \rangle = \langle \psi | \hat{O} | \psi \rangle
```

The $\hat{O}|\psi\rangle$ portion we already have a method for, so let's port it over to this use-case. For now we'll look at the single-qubit case. If we want to take advantage of threading, we need to be very careful of [race conditions](https://en.wikipedia.org/wiki/Race_condition). One option to avoid these is to use an [atomic operation](https://docs.julialang.org/en/v1/manual/multi-threading/#man-atomic-operations), which forces only one thread at a time to access the underlying data. Another option is to have each thread accumulate their own partial results (which avoids data races), then combine them at the end.
"

# ╔═╡ be3ee1db-daaa-4c20-acc9-7262f2ac9ed2
function compute_expval_shifting(O::Matrix{ComplexF64}, o_qubit::Int, ψ::Vector{ComplexF64})
	n_qubits     = Int( log2(length(ψ)) )
	temp_results = zeros(Float64, Threads.nthreads())
	Threads.@threads for ix in 0:2^(n_qubits-1)-1
		amplitude_0 = expand_int(ix, o_qubit)
		amplitude_1 = flip_bit(amplitude_0, o_qubit)
		amplitude_0 += 1
		amplitude_1 += 1
		@inbounds begin
			ψ_0 = ψ[amplitude_0]
			ψ_1 = ψ[amplitude_1]
			ix_dot_result = conj(ψ_0) * O[1] * ψ_0 + conj(ψ_1) * O[2] * ψ_0 + conj(ψ_0) * O[3] * ψ_1 + conj(ψ_1) * O[4] * ψ_1
			temp_results[Threads.threadid()] += real(ix_dot_result)
	
		end
	end
	return sum(temp_results)
end

# ╔═╡ 4b191e7e-6d28-4ab8-8d9f-1fbc9ef21e18
md"
Let's check this for correctness compared with our initial naive approach above.

Consider the rather funny looking statevector

```math
\begin{align*}
|\psi\rangle =& \frac{1}{2}|000\rangle - \frac{\imath}{2}|101\rangle + \frac{\imath}{2} |010\rangle - \frac{1}{2}|111\rangle \\
\langle \psi | \hat{Y}_0 | \psi \rangle =& \frac{1}{2}\langle \psi | \left( \imath |100\rangle - |001\rangle - |110\rangle + \imath|011\rangle\right) \\
=& \frac{1}{4}\left(\langle 000 | + \imath \langle 101 | - \imath \langle 010 | - \langle 111 | \right) \left( \imath |100\rangle - |001\rangle - |110\rangle + \imath|011\rangle\right) \\
=& 0 \\
\langle \psi | \hat{Y}_1 | \psi \rangle =& \frac{1}{2}\langle \psi | \left( \imath |010\rangle + |111\rangle + |000\rangle + \imath|101\rangle\right) \\
=& \frac{1}{4}\left(\langle 000 | + \imath \langle 101 | - \imath \langle 010 | - \langle 111 | \right)\left( \imath |010\rangle + |111\rangle + |000\rangle + \imath|101\rangle\right) \\
=& \frac{1}{4} (1  - 1 + 1 - 1) \\
=& 0 \\
\langle \psi | \hat{Y}_2 | \psi \rangle =& \frac{1}{2}\langle \psi | \left( \imath |001\rangle - |100\rangle - |011\rangle + \imath|110\rangle\right) \\
=& \frac{1}{4}\left(\langle 000 | + \imath \langle 101 | - \imath \langle 010 | - \langle 111 | \right) \left( \imath |001\rangle - |100\rangle - |011\rangle + \imath|110\rangle\right) \\
=& 0 \\
\end{align*}
```
"

# ╔═╡ 5629622e-f5f3-44fc-b43f-39cfee7da076
ψ_test = 1/2 * [1, 0, im, 0, 0, -im, 0, -1]

# ╔═╡ a905f90f-9fb3-46bc-aace-c3533cd2980c
compute_expval_shifting(ComplexF64.(Y), 0, ψ_test)

# ╔═╡ 2b144334-4f16-4d39-9c9b-9e6fb0615742
compute_expval_shifting(ComplexF64.(Y), 1, ψ_test)

# ╔═╡ 54b71b50-64b1-475b-ae88-120da3170ef4
compute_expval_shifting(ComplexF64.(Y), 2, ψ_test)

# ╔═╡ 35ae473a-d80a-4371-b4ab-c5fda3041693
md"
They match! It would of course be good to check `X` as well, let's do that quickly...
"

# ╔═╡ 35a81740-63e0-49a6-82c4-10fe7093e626
md"
And we can compare performance too...
"

# ╔═╡ af44c952-82dd-4d18-8c94-225fb3a2790a
expval_shifting_runner(ψ) = compute_expval_shifting(X_float, 2, ψ)

# ╔═╡ 330e6b2c-3d23-49da-8395-57f336b99f56
@b ψ_big() expval_shifting_runner

# ╔═╡ 9e373493-20c3-4d46-b65b-ff565bf6c355
md"
Now that is **much** better, even for a relatively small problem! Here we can see the benefit of using these in-place bitshifting operations, even for a relatively small statevector with only 12 qubits.
"

# ╔═╡ 42df3582-9bd1-4676-b25e-32df9ad15d7b
md"
## Sampling

Computing exact expectation values is all well and good, but a real quantum computer runs the same circuit many times and then measures a single output bitstring -- a \"shot\". From these shots we can compute (estimated) expectation values and correlators, but on the real quantum hardware we don't have access to the full statevector. So we would also like a way to model what the output bitstring distribution for a given shot count will look like, especially once we're ready to add noise into the mix.
"

# ╔═╡ ac05e3aa-3327-48ad-be4c-8aa485cad85f
md"
### Naive approach to sampling

We know that the elements of the statevector represent probability *amplitudes* and the actual probability of measuring a given state is 
```math
p_{\phi} = || \langle \phi | \psi \rangle ||^2
```

Assuming $|\psi\rangle$ is normalized. A common and simple approach to sampling from the vector of \"weighted probabilities\" is to take a cumulative sum of all of them, `summed_p`, normalize this new vector, then compute a random number `needle` in `[0, 1)`. Happily, Julia's [`rand`](https://docs.julialang.org/en/v1/stdlib/Random/#Base.rand) can generate such a random number for us. Then one picks the first element of `summed_p` which is larger than `needle` and that index corresponds to the sampled bitstring for the shot. Let's implement this as a quick testing utility:
"

# ╔═╡ 20d99560-3f8e-42b9-9a1a-ebbe86833620
function naive_sample(ψ::Vector{ComplexF64}, n_shots::Int)
	probabilities = abs2.(ψ)
	summed_p = cumsum(probabilities)
	summed_p ./= summed_p[end]
	shots = map(1:n_shots) do shot_ix
		needle = rand()
		ix = findfirst(p->p > needle, summed_p)
		return ix
	end
	return shots
end

# ╔═╡ 4160f4ad-88af-4fab-8c82-f978e80e1e0e
md"
If you're not familiar with the [`cumsum`](https://docs.julialang.org/en/v1/base/arrays/#Base.cumsum) function, it implements the common [\"prefix sum\"](https://en.wikipedia.org/wiki/Prefix_sum) or \"scan\" operation.
"

# ╔═╡ 6a4da26f-7243-48b6-b1d6-99748fee1aa7
cumsum([1, 4, 5, 3, 10])

# ╔═╡ 3475207b-57fe-499c-9311-16c31028dfae
md"
Some easy tests for this are GHZ or \"cat\" states, which are an even linear combination of \"all 0\" or \"all 1\".
"

# ╔═╡ 1bac4307-03a3-4242-b6fb-6703b5abf165
begin
	ψ_ghz = zeros(ComplexF64, 2^5)
	ψ_ghz[1] = one(ComplexF64)
	ψ_ghz[end] = one(ComplexF64)
	naive_sample(ψ_ghz, 4)
end

# ╔═╡ 19872326-3eba-43d6-b897-9fb387828a8e
md"
Another test is to sample over a state vector with equal probability amplitudes for all bitstrings.
"

# ╔═╡ f9fc1eb0-2002-4ef4-8e28-e7b02292dae1
begin
	ψ_even = normalize!(ones(ComplexF64, 2^5))
	naive_sample(ψ_even, 10)
end

# ╔═╡ 5bc669a3-44da-4920-834d-164393b65d1d
md"
OK, given that we called this \"naive\", there must be some problems with it. And there are! For very low shot counts, it's not in fact so bad, but for larger numbers of shots much better algorithms can be used. The main technique we'll look at is the use of an [alias table](https://en.wikipedia.org/wiki/Alias_method), which allows us to construct an efficient lookup table in $\mathcal{O}(n)$ (with a small prefactor), then do lookups in $\mathcal{O}(1)$. This is substantially better than having to do a $\mathcal{O}(n)$ linear search *for each shot*.

Several implementations of the alias table technique exist in the Julia ecosystem. For simplicity, we'll use [`AliasTables.jl`](https://aliastables.lilithhafner.com/dev/).
"

# ╔═╡ 539b498b-a8bb-4807-8609-84df0e34552b
function alias_sample(ψ::Vector{ComplexF64}, n_shots::Int)
	at = AliasTable(abs2.(ψ))
	return rand(at, n_shots)
end

# ╔═╡ a65a07e4-11b5-4459-9f46-750dee2125b7
md"
Let's do some quick sanity checks then compare performance again.
"

# ╔═╡ 26044735-c046-4d4c-b14d-f4556ed269b6
alias_sample(ψ_ghz, 4)

# ╔═╡ c2431a78-2b73-4cae-8ae2-a9f29f5a49a3
alias_sample(ψ_even, 10)

# ╔═╡ 6cdf0d4c-9c27-47af-8684-e142383727f4
@b normalize!(rand(ComplexF64, 2^14)) (ψ->naive_sample(ψ, 20))

# ╔═╡ 21d7edad-6477-46b8-ae82-95796c90be64
@b normalize!(rand(ComplexF64, 2^14)) (ψ->alias_sample(ψ, 20))

# ╔═╡ 42fac6d3-5d1c-4608-a9e2-bac922d9d663
md"
So for relatively few shots, the naive approach seems to do better. How about for a very large number of shots?
"

# ╔═╡ 9027b4e7-fdca-408d-b3db-2fa57cdf91e1
@b normalize!(rand(ComplexF64, 2^14)) (ψ->naive_sample(ψ, 500))

# ╔═╡ 90ba2d81-3736-470a-8151-72d52e92826a
@b normalize!(rand(ComplexF64, 2^14)) (ψ->alias_sample(ψ, 500))

# ╔═╡ b89072ad-a499-4837-9325-c9410c25b6e5
md"
As expected, the alias table method works much better here. The specific crossover point is of course pretty dependent on the state vector size and your hardware, but even at low shot counts the alias table is not so much worse."

# ╔═╡ 0c591cc5-3d03-4038-a002-1b447de4b628
md"
## Do we really have to care about performance so much?

For prototyping or correctness checking, for example to compare against tensor network algorithm, the quick and dirty \"naive\" approach is perfectly fine. However, for larger system sizes or for large collections of circuits, these fixes can make a huge difference in what's even feasible to simulate - a 10-fold difference can take something from a month's runtime to a few days.
"

# ╔═╡ b42a1b2f-a49d-4f3f-a2e3-715ff3e9cd47
md"
## Density matrices and noise simulation

If we want to simulate the effects of (Markovian) noise on a quantum system, one approach is to use a density matrix to describe the resulting mixed state. A state vector, describing a pure state, can of course form a density matrix

```math
\hat{\rho} = | \psi \rangle \langle \psi |
```

But for mixed states, such a decomposition into a single outer product is impossible. Such mixed states can be generated by noise in a quantum computer, in particular so-called Kraus noise, which can also be described as a set of CPTP operations.

Let's remind ourselves of some basic facts about density matrices. Gate application, analogously to the pure state case, can be implemented as follows:

```math
\hat{\varrho} = \hat{U} \hat{\rho} \hat{U}^\dagger
```

Where $\hat{U}$ is some unitary. Expectation values can be computed with the following formula:

```math
\langle O \rangle = \mathrm{Tr}\left(\hat{O}\hat{\rho}\right)
```

Is there a way we can implement these operations without having to write a lot of new code? Again, let's take the approach we used before, and write a naive implementation first to guide us.
"

# ╔═╡ f5595113-b9d0-4764-884c-095126ae161e
function apply_gate_naive(ρ::Matrix{ComplexF64}, gate::Matrix{ComplexF64}, gate_qubit::Int)
	n_qubits  = Int( log2(size(ρ, 1)) )
	all_gates = [I for qubit in 1:n_qubits]
	all_gates[n_qubits - gate_qubit] = gate # Julia is one indexed!
	full_gate = reduce(kron, all_gates)
	return full_gate * ρ * adjoint(full_gate)
end

# ╔═╡ 7eebb5b3-a896-4bf6-9efd-0e609b18660c
ϕ = apply_gate_naive(ψ, X, 1)

# ╔═╡ 83a5f7fa-1ee4-4b9e-beac-5fb748fc0770
ϕ_naive1 = apply_gate_naive(ψ3, X, 0)

# ╔═╡ e2fea3a9-5a30-44ce-ae78-90a3eab881dc
ϕ_naive2 = apply_gate_naive(ψ3, X, 1)

# ╔═╡ 472edfe1-99cb-4ffa-90cf-aef843453a1f
ϕ_naive3 = apply_gate_naive(ψ3, X, 2)

# ╔═╡ d00c1c83-b8f3-4c66-9b06-3dda6dcf4d9a
naive_runner(ψ) = apply_gate_naive(ψ, X, 2)

# ╔═╡ 35e08ed1-2edd-43ed-b8de-9c3d03b5964c
@b ψ_small() naive_runner

# ╔═╡ 25c415dd-bf22-408d-b517-db4f444640c8
@b ψ_big() naive_runner

# ╔═╡ 34473376-1f23-493c-95b5-b1c4db8c9fb3
@b ψ_small() naive_runner

# ╔═╡ 00e9df7b-500e-40d1-8f31-0055dffc9dce
@b ψ_big() naive_runner

# ╔═╡ c80b0982-5a86-49be-8b5a-449e3ab7ddc9
apply_gate_naive(ψ3, X, 0)

# ╔═╡ dce58da5-297f-426b-a26c-4561fba84385
apply_gate_naive(ψ3, X, 1)

# ╔═╡ 302b63b5-b5c7-48b1-8809-1dca01269a83
apply_gate_naive(ψ3, X, 2)

# ╔═╡ 36eb7778-d19a-42ec-bbe8-91fb47cffb00
md"
We can quickly check this by constructing a density matrix from the pure states we used above.
"

# ╔═╡ 29abe7a9-6c60-4b32-957d-85d701033472
ρ_basic = kron(adjoint(complex.([1/√2, 0, 0, -1/√2])), [1/√2, 0, 0, -1/√2])

# ╔═╡ 84b1da4d-e977-4b70-8e21-ea7df1ad72bd
apply_gate_naive(ρ_basic, X_float, 1)

# ╔═╡ af0accb2-223d-432f-a01d-e8bdd561a763
md"
Is this right? We can check either by construction the density matrix corresponding to the correct (pure) state vector or writing a quick routine to compute the [partial trace](https://en.wikipedia.org/wiki/Partial_trace) in order to find the [Schmidt decomposition](https://en.wikipedia.org/wiki/Schmidt_decomposition) of the output density matrix. The first is easiest, so let's do that for now.
"

# ╔═╡ a71303ce-25cb-4bc5-98fa-ed3fd48ae36c
correct_ρ_output = kron(adjoint([0.0+0.0im, -0.707107+0.0im, 0.707107+0.0im, 0.0+0.0im]), [0.0+0.0im, -0.707107+0.0im, 0.707107+0.0im, 0.0+0.0im] )

# ╔═╡ a94215d6-90ba-4e4f-b623-a52f7e86ca7c
md"
Up to some sign changes for zeros, this looks good! Hooray. Again, it has all the problems of our previous super-naive approach for state vectors, but we'll encounter those even earlier as matrix-matrix multiplication has worse scaling than matrix-vector, and the density matrix representation of an `N` qubit state has size $2^N \times 2^N$.

We could in fact have used our previously defined `apply_gate_naive` for state vectors to implement this. How? Let's think a little more about the fact that we have a (Hermitian) matrix -- we could, in principle, reshape this into a $2^{2*N}$ *vector*, which is how it's stored in memory. This \"vector\" isn't a proper state vector, of course -- for one thing, it's not normalized -- but it can save us having to write custom methods for density matrices.
"

# ╔═╡ 9b1f9abe-4a07-4feb-bea8-1b36d2f2675b
reshaped_ρ_basic = reshape(ρ_basic, 16)

# ╔═╡ 76ce6d2e-8d42-4c58-9edd-cec464b4b4a4
begin
	ρ_intermediate = apply_gate_naive(reshaped_ρ_basic, X_float, 1)
	reshape(apply_gate_naive(ρ_intermediate, Matrix(adjoint(X_float)), 1 + 2), (4, 4))
end

# ╔═╡ c8a3eacb-893a-4750-92cb-876bfa3c34f0
md"
Why add the number of qubits represented by $\rho$ to the second gate application? When we compute

```math
\varrho = \hat{U}\hat{\rho}\hat{U}^\dagger
```

Or, if we write the full Einstein summation

```math
\begin{align*}
\varrho_{i,j} &= \sum_{k,\ell} U_{i,k} \rho_{k,\ell}  U^\dagger_{\ell,j} \\
&= \sum_{k,\ell} U_{i,k} \rho_{k,\ell} U^\dagger_{\ell,j} \\
&= \sum_\ell \left(\sum_k U_{i,k} \rho_{k, \ell}\right) U^\dagger_{\ell, j} \\
\end{align*}
```

The inner summation here looks quite a bit like a matrix-vector multiplication, of course. If we reshape $\rho$ to be a $2^{2N}$ vector $\vec{\rho}$, we're effectively doing:

```math
\vec{\rho}_{k+\ell \times 2^N} = \hat{\rho}_{k,\ell}
```

Let's quickly check that is actually the case:
"

# ╔═╡ 2b9630d8-c2fb-488c-9d84-18d1e5b7cdc9
sample_matrix = [1 2; 3 4]

# ╔═╡ 2a15d869-432a-48fb-9d81-a72c8eb67e6f
reshape(sample_matrix, (4,))

# ╔═╡ 9ab0965a-181d-4a3e-84d9-746485bc070d
md"
So Julia does indeed go \"down the columns\" first (column-major ordering). Let's call the result of the inner summation $\rho'$. Then:

```math
\begin{align*}
\rho'_{i,\ell} &= \sum_k U_{i,k}\rho_{k + \ell \times 2^N} \\
\varrho_{i,j} &= \sum_\ell \rho'_{i,\ell}U^\dagger_{\ell,j} \\
\end{align*}
```
But, due to the reshaping, every $\ell$ is $2^N$ apart, effectively shifting the $\ell$ indices over by $N$ qubits. We can prove this to ourselves using Julia's [`CartesianIndices`](https://docs.julialang.org/en/v1/base/arrays/#Base.IteratorsMD.CartesianIndices) and [`LinearIndices`](https://docs.julialang.org/en/v1/base/arrays/#Base.LinearIndices). Returning to `ρ_basic`, we can examine a pair of indices at a time:
"

# ╔═╡ 07392403-40fd-49cf-9ca6-ed70c6909f6a
for ind in LinearIndices(ρ_basic)
	println(ind, " ", bitstring(ind-1))
end

# ╔═╡ b67925e3-4020-49d8-9872-e3980eb3ad73
for ind in CartesianIndices(ρ_basic)
	println(ind.I, " ", bitstring(ind.I[1]-1), " ", bitstring(ind.I[2]-1))
end

# ╔═╡ 8622417c-7642-404a-9405-ef0fab2bf62f
md"
This is a bit of a pain to look at, but hopefully you can see that the linear indices above have the leftmost `CartesianIndex`'s component's bits spliced in to the left 2 bits, and the rightmost `CartesianIndex`'s component's bits spliced in to the right 2 bits. So, by treating the density matrix as a \"vector-like\" object, we've effectively created a `2*Q` \"state vector\" (not normalized!) which we can reuse a lot of our logic on.
"

# ╔═╡ 19146efc-ded9-4ed7-9a5b-46031d736d27
md"
Along similar lines, we can write a naive expectation value function.
"

# ╔═╡ cc8950cc-22be-494c-b55e-9ba19571cfb6
function compute_expval_naive(O::Matrix, o_qubit::Int, ρ::Matrix{ComplexF64})
	n_qubits = Int( log2(size(ρ, 1)) )
	all_observables = [ComplexF64.(I) for qubit in 0:n_qubits - 1]
	all_observables[o_qubit + 1] = O
	full_O = reduce(kron, all_observables)
	return sum(diag(full_O*ρ))
end

# ╔═╡ 4fb3c3e3-9ec4-4d95-86f4-9b70ba540e5f
compute_expval_naive(Z, 0, ϕ)

# ╔═╡ d379cd07-289e-4baf-8159-547fc27311cf
compute_expval_naive(Y, 0, ψ_test)

# ╔═╡ 31d5c8f6-f23c-42b1-9618-24baf4b7a0d5
compute_expval_naive(Y, 1, ψ_test)

# ╔═╡ 4af90965-67f8-4348-adef-557f1dcf6c62
compute_expval_naive(Y, 2, ψ_test)

# ╔═╡ 805bac9c-2925-4028-b457-9f40cbc912cf
for qubit in 0:2
	println("Qubit $qubit - Naive expval: $(compute_expval_naive(X_float, qubit, ψ_test)), shifting expval: $(compute_expval_shifting(X_float, qubit, ψ_test))")
end

# ╔═╡ b803b71f-7559-42c7-b0c7-356abb801f98
expval_naive_runner(ψ) = compute_expval_naive(X_float, 2, ψ)

# ╔═╡ f55c53c9-ab8e-4338-9993-83310a24201c
@b ψ_big() expval_naive_runner

# ╔═╡ b24ab29b-57ab-4eba-b81e-f5095af74c1c
compute_expval_naive(ComplexF64.(I), 1, ρ_basic)

# ╔═╡ 68ce341f-c14f-4a80-b8c5-cdcdd795207e
compute_expval_naive(ComplexF64.(I), 1, complex.([1/√2, 0, 0, -1/√2]))

# ╔═╡ b06c0a01-b3ac-48d6-b77a-9c3c630627a0
md"
### Handling Kraus noise

Will the trick of treating the density matrix like a vector work for Kraus noise? Again, we can test this out. Here we'll only look at Markovian noise, which maintains the unitarity of the overall density matrix. Another way of saying this is that the trace of $\rho$ will remain $1$. There are a variety of common \"types\" of noise (see, for example, the [Qiskit list](https://docs.quantum.ibm.com/guides/build-noise-models#quantum-errors)), such as bit or phase flips, or depolarizing errors. These can be written in a Kraus map form. Kraus maps need to be *completely positive trace-preserving* (CPTP), which means they keep the trace valued at 1, **and** all eigenvalues of the output density matrix are greater than or equal to 0.

For bit flips, we say that a flip occurs with probability $p$:

```math
\hat{\varrho} = (1 - p) \hat{\rho} + p (\hat{X}\hat{\rho}\hat{X})
```

Many of the noise channels are sums of operations like this. This will be difficult to implement with `apply_gate_naive` because we can't write the operation as a single matrix. We could copy $\hat{\rho}$ and then apply $\hat{X}$ to the copy, then write something more sophisticated.
"

# ╔═╡ 6644d2b7-cd07-4476-a13e-4abb7031bab0
md"
### Exercises

1. Try writing an equivalent of `apply_gate_naive` which can handle *sums* of operators, like bit flips
2. Try using the bitshifting approach to write a less naive kernel, as we did above for gate applications, to implement these Kraus maps
"

# ╔═╡ c0370e95-65db-47ed-8f22-a205c80ede2d


# ╔═╡ 95a128f6-bbf0-49cc-bdfb-c427682a3b03
md"
# Summary (so far)

- Optimizing the inner \"hot\" functions where our simulator will be spending a lot of time can be worth doing
- Some quick-and-dirty unoptimized experimentation is perfectly fine for testing and checking correctness
- Profiling is the best way to find out what to target in optimization
- Direct benchmarking will help us figure out when each technique makes sense to use
- Avoiding allocations and doing operations in-place can deliver large speedups for large arrays (high qubit count)
- Ensuring we don't need to do type conversion or promotion can also speed up the inner gate/noise application
- Threading can also be worthwhile but we need to be careful to avoid data races
"

# ╔═╡ 20006671-24b7-4210-877a-9ed401ec350a
md"
## Running many gates in sequence

For a real circuit simulation, we probably would like to apply many gates, one after another. This can introduce even more performance considerations -- negligible contributions for 1-10 gates can become serious drags when 10,000 are applied.

For a somewhat cartoonish example, let's examine the following scenario:
"

# ╔═╡ cfbdfe7a-20e4-4a77-8d3e-33ffe9535213
X_generator() = [zero(ComplexF64) one(ComplexF64); one(ComplexF64) zero(ComplexF64)]

# ╔═╡ f7c71bce-98e3-47a5-936f-31c367a9208c
ψ_circuit = normalize!(rand(ComplexF64, 8))

# ╔═╡ d7dfd004-69e2-42dd-af39-a7620faa706d
@time for ii in 1:10
	apply_gate_shifting_linear(ψ_circuit, X_generator(), 1)
end

# ╔═╡ 4ed254a6-2960-41b3-9d59-947abe27b597
@time for ii in 1:10_000
	apply_gate_shifting_linear(ψ_circuit, X_generator(), 1)
end

# ╔═╡ 3df62f81-3a75-4c8d-b93b-0e002f81eeb6
md"
We can see that the latter run is worse than 100x slower, as we might have expected. `@time` explains why -- we're spending time in garbage collection (and in array allocation). This example creates a large number of small, short-lived arrays which all have to be cleaned up, which is just about the worst-case scenario for Julia's GC. Obviously this is quite contrived, since we could and did pre-allocate an `X` above, but it can be more of a problem once we add parameterized gates into the mix. Such gates would be the Pauli rotations `RX`, `RY`, `RZ`; the phase shift gate; the 3-angle arbitrary unitary gate, and many others. For these, each time we input a new angle or parameter, we might have to create a new matrix representing the gate. You can see how this could quickly induce a lot of allocations if we're not cautious. Is there a way to avoid doing so?

Julia's GC handles *heap* allocations, while small, short lived objects are *stack* allocated. If you haven't heard of the stack and the heap before, [here's](https://stackoverflow.com/questions/79923/what-and-where-are-the-stack-and-heap) a quick summary. Small, short-lived objects are exactly what we need to implement many gates, so finding a way to put them on the stack rather than the heap can save us a lot of GC pressure. It's helpful that most quantum gates are on only a few qubits due to current hardware limitations, so their total size is quite small.

One option to avoid heap allocation is to create a new `apply_gate` method for each gate type, then generate the elements on the fly from the parameters. Another is to represent the gate as a *tuple*, which is stack allocated. In fact, Julia provides a nice package for representing small arrays -- [`StaticArrays.jl`](https://github.com/JuliaArrays/StaticArrays.jl) -- which uses tuples \"under the hood\" to make sure everything is stack allocated. `StaticArrays.jl` also implements may high performance functions for small array operations, so it's a great choice for our gates.

A few more tips to avoid GC pressure holding your performance back:

- If you're running multiple simulations *and* need to conduct sampling, you can preallocate and [reuse](https://aliastables.lilithhafner.com/dev/#AliasTables.set_weights!) your alias table(s)
- If you need to make a vector of indices, see if you can use an `SVector` (one of the vector types in `StaticArrays.jl`), which is also stack allocated
- Try to use a [generator](https://docs.julialang.org/en/v1/manual/arrays/#man-generators) rather than an array, if you can
"

# ╔═╡ 09dc2f34-ed91-4046-8b3c-2a88119aa7a6


# ╔═╡ 3dc53acf-9b5d-438e-8882-3dfd0fd3651a
md"
## A few more exercises

- Try writing a function to compute a partial trace for a state vector or density matrix. What if the qubits are not in order?
- Can you speed up simulations by combining some gates in certain circumstances? (Hint: it can be very expensive to loop over the entire state vector...)
- Is bitshifting always the most effective technique? Could you achieve something similar but perhaps more efficient with other operations?
- Modern hardware doesn't read memory byte by byte, but in fact as a series of \"words\", which have pre-defined [sizes](https://en.wikipedia.org/wiki/Word_(computer_architecture)) depending on the architecture. Is there a way to use this fact to improve your memory access pattern? 
"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AliasTables = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
Chairmarks = "0ca39b1e-fe0b-4e98-acfc-b1656634c4de"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Profile = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[compat]
AliasTables = "~1.1.3"
Chairmarks = "~1.3.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.11"
manifest_format = "2.0"
project_hash = "1860f456c40350cb0ccfc48165b55c8a119b335a"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Chairmarks]]
deps = ["Printf", "Random"]
git-tree-sha1 = "9a49491e67e7a4d6f885c43d00bb101e6e5a434b"
uuid = "0ca39b1e-fe0b-4e98-acfc-b1656634c4de"
version = "1.3.1"

    [deps.Chairmarks.extensions]
    StatisticsChairmarksExt = ["Statistics"]

    [deps.Chairmarks.weakdeps]
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+5"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"
"""

# ╔═╡ Cell order:
# ╟─42b097f0-4c69-11f0-39d8-55affea6301b
# ╟─2ee77de2-ea9b-4d1e-9af7-7acfe0b6af10
# ╟─f2ae4404-8f42-4078-8079-d46c4bacabfe
# ╟─66e92550-586d-4726-9478-7097ae294ec6
# ╟─c0e6e8ee-9ce6-4f5d-b5c6-e964eb9d8f50
# ╠═9a82d5ec-c36a-4a96-9c6e-70c344179801
# ╠═2331d6d7-7c09-4fa4-b6e5-a9a5aa144a1a
# ╠═462bbc01-26a1-4f9c-8101-4e35bdb29084
# ╠═811f0d69-d1d7-44fb-9e83-9f1fc434fd95
# ╠═c21d83bf-fa20-424e-b9cf-3171c6162151
# ╠═ade1bd46-f929-47b5-a3f3-47104a64c177
# ╟─e3f1bbbc-b77f-415e-821d-7049d000ff0a
# ╠═895b29ab-fd4c-48f0-b34c-5faa7bbc4ecc
# ╠═7eebb5b3-a896-4bf6-9efd-0e609b18660c
# ╟─c224a2f3-3960-4247-a605-3439d618ca24
# ╟─2a2039e3-3a6b-49ac-8737-5e97b6fcb935
# ╟─a52b8837-cd46-4452-a6eb-3d922411ff9e
# ╠═ee03bbbe-cfb4-4d11-a544-8e0ba30aba11
# ╟─7a654e8c-7abe-4623-a1db-fa99eb5bdb6b
# ╠═01af6d1f-f4ca-4cc9-8fb3-61f1b0242848
# ╠═21fd60c6-8b65-4803-a920-6da39a4b3cfd
# ╟─87826bb8-e120-4615-ba82-0d1e6fbdf0b6
# ╠═4fb3c3e3-9ec4-4d95-86f4-9b70ba540e5f
# ╟─43e463e6-a9ea-4528-b2b1-4c2a9327c593
# ╠═d38909f3-7b4d-4a6f-8e1c-c939203d587a
# ╟─da55b2ed-b560-4ce6-a79f-2e1735e2500c
# ╟─948fa3b6-be17-4f5d-a680-d4f4fecca8e8
# ╠═9ea2a657-a6bd-44ea-ad88-133fd8906bae
# ╠═99bc23ab-3327-4668-8252-eb4ff080c18b
# ╠═cdc44a67-b573-4fb2-a68f-1c7b5d62958e
# ╠═45d1b011-c2ed-4c8f-aa29-cba8bf24bb67
# ╠═29da18fe-0935-4cb0-8f99-9791bce298a5
# ╟─25bbcd81-43e4-4de8-a6bf-a2d3456e72c1
# ╠═9165bec9-c0ee-4606-aafa-ffe1f6eb32b5
# ╠═113037cb-6431-431d-af3a-840e5fd47473
# ╠═83a5f7fa-1ee4-4b9e-beac-5fb748fc0770
# ╟─2eaf59b7-fa3b-47f4-b662-fbf4183e0ee5
# ╠═e2fea3a9-5a30-44ce-ae78-90a3eab881dc
# ╠═2f50fb28-a328-4a30-b048-8b901738624c
# ╟─0cd36d05-41fb-441a-9307-6367e53a721e
# ╠═93a3be9d-ddef-473a-8997-78091b176275
# ╠═06052b8c-03a7-47c8-b8cd-a122db0e0bea
# ╟─18d097b3-a9ed-4910-a201-e48d5a836f6b
# ╠═472edfe1-99cb-4ffa-90cf-aef843453a1f
# ╠═510bae1b-cd56-4844-aab7-cbf1a23b31c2
# ╟─d9a62485-330e-4367-999e-44ea2d75397e
# ╟─e13154d5-e3a9-46eb-836d-8cbe15630002
# ╠═60ec9535-59d1-4545-af77-4b21d0bb811a
# ╟─18b15e49-b733-4104-a031-42b7b8a53acc
# ╟─c87c7cd5-7234-4006-8a54-2510a59dbeb7
# ╠═57ed8099-7ede-4bb8-9a28-2945083baacf
# ╠═9e9d8943-57ef-47fa-90fd-81d69f346bd8
# ╠═d00c1c83-b8f3-4c66-9b06-3dda6dcf4d9a
# ╠═e67e0a89-d3ab-431a-a640-9c9bf40acf3d
# ╠═35e08ed1-2edd-43ed-b8de-9c3d03b5964c
# ╠═c2c830fa-f6f6-4b70-b487-4913f93bdd04
# ╟─fd2f0aea-06cc-41a2-a781-6f3b4b126e11
# ╠═186daf87-d823-4907-93ee-ad82ec667f70
# ╠═25c415dd-bf22-408d-b517-db4f444640c8
# ╠═6da0833a-c1ef-46d5-826e-f8b26d0da783
# ╟─9e8b447e-7e3e-4d99-890e-9140388153ee
# ╟─2510c7f7-55b8-49e8-bc0e-ce59952a4079
# ╠═af671a1c-dd98-4046-a063-5e6a42e1174f
# ╟─3bbc13c8-ff21-4404-8600-f3922f65c961
# ╠═77f4ca16-7fed-4025-b9b8-a00c06eb1d5f
# ╠═71bc7a01-7971-4443-9ee9-db2dd289bcaa
# ╟─3e148898-4ec3-424b-8885-7689109512ba
# ╠═d7bfaa44-f577-4487-8de6-77b137871d9c
# ╟─8053adfd-cddf-4b3d-9a0f-fcb2bd23e42c
# ╠═b7532daa-a526-421a-8db9-4c4539c0c969
# ╠═2d9f1ca6-0855-4104-9d5a-c8bd51880dff
# ╟─afce6fe8-7876-4766-899c-a3bc8a35451e
# ╠═0fb793b9-b3cc-4d97-b6e6-5407ab3c23c2
# ╠═afafd52e-72b4-4a7b-8cc5-f7fbbf218c39
# ╟─7ea1ed14-dab6-47ae-970c-7dee8b62ca1a
# ╠═49d94720-e428-41c8-a5a6-035a4472c555
# ╠═c80b0982-5a86-49be-8b5a-449e3ab7ddc9
# ╠═ae670d56-2979-40a6-b71a-b49c15ef6613
# ╠═dce58da5-297f-426b-a26c-4561fba84385
# ╠═407f0764-1388-454a-b132-8d28130a18c4
# ╠═302b63b5-b5c7-48b1-8809-1dca01269a83
# ╟─fb2f9abe-0f6f-4c58-a46e-90e339d8ac98
# ╠═6f8f5a92-2421-4f89-b165-010db2ef6f98
# ╠═34473376-1f23-493c-95b5-b1c4db8c9fb3
# ╠═a7df0f04-d43d-42e7-afdf-6da9984a30bc
# ╠═aebc17c3-f762-477e-a26a-519c701ffcb9
# ╠═4f78fb80-6b17-4f5f-b47d-e5d55aaa7764
# ╟─2b5428cc-8890-42b3-91e9-d3b5801ac2d8
# ╠═00e9df7b-500e-40d1-8f31-0055dffc9dce
# ╠═4008b027-96f5-4b2f-beeb-de4a5fa524c9
# ╠═299ddda6-b7b1-4d72-9a2c-4aac5eefaf8c
# ╟─e8fb90e4-f741-4125-9e67-1ea9d340c018
# ╠═12c26fb0-0c83-4beb-a367-3bbfa635ae61
# ╟─40c345ea-6b3e-4fa1-a60c-b6abb2ef4e5a
# ╠═0c261603-48ee-4b5a-8727-02d3623203f5
# ╠═6fe606a8-d91a-4b00-bd1c-a352623c3985
# ╠═0efb844f-7575-426d-b5f5-ee235b7f16e4
# ╠═77bd3384-c224-45ba-acab-32dd01cedab1
# ╟─a016e2cb-1b5c-46c9-bfae-f35df2a294d2
# ╠═700ef4ad-6053-45c0-a8ff-b8494d8be576
# ╠═0a239d26-796d-4721-b2f9-c77b8c64b354
# ╠═bab81d91-c3c1-44a0-8dee-2022b5df785d
# ╠═fabf4d37-ecaa-40e4-bc18-23c928b48cbe
# ╠═c3efeba8-722e-4962-b072-e2d8212e40a0
# ╠═189f5812-a095-4371-b6e8-fe5bf5a9af5c
# ╟─c4cb5ee6-ae5f-4918-a485-4e7ae8ccfc8c
# ╠═5afc29e6-c4ec-4c3d-b080-051eb7caa1db
# ╠═f99154d6-0b48-484c-a36c-2f2b4e099e1c
# ╠═7c2ede52-bcc6-4c1c-bc56-ebc9ca37b126
# ╟─5f0eb332-d979-4df1-b57f-24e1e2b6680b
# ╠═040f885a-8ddb-4254-978a-bc7420ec7877
# ╠═eb099f55-a8b4-48f3-87cb-644330b150e9
# ╠═d18e277f-ecf6-4d4f-a6d4-f8404d0df59b
# ╟─bad3f02f-9685-476b-9ae3-7c455b91b993
# ╠═6a2f6927-8a01-40d9-a512-0f75d4f2520a
# ╠═a2d44668-bbbd-4963-b283-0b0929e2bf00
# ╠═dbf2ef8b-afce-4adb-9999-312a6a5eb7a0
# ╠═af0e7a78-231c-48e7-b116-356ef92dee28
# ╠═d8ce2c16-0556-4c7d-bafe-a6efaa46a1a0
# ╟─5790bb0f-1f6a-4278-bce5-87fc0bb33a92
# ╠═228b7313-ec14-4f2d-8c69-f9f6de909a10
# ╠═58595934-5389-434a-9f33-538d4b71c31e
# ╟─47ad6221-e5bd-4f77-b8ad-28231d30b315
# ╠═13a83ffa-666f-42f6-9f62-a8088f512bc9
# ╟─456976ea-7e5f-4cc0-ad5e-1bfef8b1da01
# ╠═479df144-24f9-4966-b2a6-39c377ee4af8
# ╠═b11f1f4c-04a9-43b9-af1c-3e5bc560dd06
# ╠═242a2d57-2478-4177-8036-a3683ecb947d
# ╠═e4691b67-1d63-49df-960e-954806c49970
# ╟─4a4d7a98-e1b0-432f-acf5-00a76a05eb6d
# ╠═cf13f425-c34c-4d6f-9605-e57f677414c9
# ╟─faf6d35f-55de-477b-b7c0-d01497776f43
# ╠═897b47c1-6bd7-4fdb-ac53-2506a0a9d959
# ╠═91cfbca9-d11d-43b7-bd9a-892a66f0db21
# ╠═84d88d31-fa73-4650-bb31-8f0d040e5ece
# ╟─c59dde08-79c3-472e-bd6d-5683b38521fe
# ╟─2a70ea35-6302-4748-85f5-4616899d5082
# ╠═2597cc50-8fc7-4363-9fdd-da2e27bc3d09
# ╟─2c4202aa-230b-44e5-a323-a62788a55212
# ╠═02645015-f490-4251-935d-0c076c5b40d5
# ╟─c2a87355-a575-4732-8dee-3a9b6ca58eed
# ╟─8d604aa0-a871-424e-a702-1c4dde4ee978
# ╠═4e36b069-5d4b-416c-9b5d-baad75e40f01
# ╟─f26c5049-d7be-49f1-9f9e-5e9f91230b0f
# ╠═44e4179a-d141-412c-8791-9e35b268f6c2
# ╟─02145b62-1bd2-4f74-873b-48bce6a76aa2
# ╠═8050906c-95fc-47dc-a755-5d1ec354fd41
# ╟─f692016d-ae34-4c86-b5ab-e8d718c6603c
# ╠═be3ee1db-daaa-4c20-acc9-7262f2ac9ed2
# ╟─4b191e7e-6d28-4ab8-8d9f-1fbc9ef21e18
# ╠═5629622e-f5f3-44fc-b43f-39cfee7da076
# ╠═d379cd07-289e-4baf-8159-547fc27311cf
# ╠═31d5c8f6-f23c-42b1-9618-24baf4b7a0d5
# ╠═4af90965-67f8-4348-adef-557f1dcf6c62
# ╠═a905f90f-9fb3-46bc-aace-c3533cd2980c
# ╠═2b144334-4f16-4d39-9c9b-9e6fb0615742
# ╠═54b71b50-64b1-475b-ae88-120da3170ef4
# ╟─35ae473a-d80a-4371-b4ab-c5fda3041693
# ╠═805bac9c-2925-4028-b457-9f40cbc912cf
# ╟─35a81740-63e0-49a6-82c4-10fe7093e626
# ╠═af44c952-82dd-4d18-8c94-225fb3a2790a
# ╠═b803b71f-7559-42c7-b0c7-356abb801f98
# ╠═330e6b2c-3d23-49da-8395-57f336b99f56
# ╠═f55c53c9-ab8e-4338-9993-83310a24201c
# ╟─9e373493-20c3-4d46-b65b-ff565bf6c355
# ╟─42df3582-9bd1-4676-b25e-32df9ad15d7b
# ╟─ac05e3aa-3327-48ad-be4c-8aa485cad85f
# ╠═20d99560-3f8e-42b9-9a1a-ebbe86833620
# ╟─4160f4ad-88af-4fab-8c82-f978e80e1e0e
# ╠═6a4da26f-7243-48b6-b1d6-99748fee1aa7
# ╟─3475207b-57fe-499c-9311-16c31028dfae
# ╠═1bac4307-03a3-4242-b6fb-6703b5abf165
# ╟─19872326-3eba-43d6-b897-9fb387828a8e
# ╠═f9fc1eb0-2002-4ef4-8e28-e7b02292dae1
# ╟─5bc669a3-44da-4920-834d-164393b65d1d
# ╠═d1b99d47-a25a-4e24-a3fa-a2b3023cd194
# ╠═539b498b-a8bb-4807-8609-84df0e34552b
# ╟─a65a07e4-11b5-4459-9f46-750dee2125b7
# ╠═26044735-c046-4d4c-b14d-f4556ed269b6
# ╠═c2431a78-2b73-4cae-8ae2-a9f29f5a49a3
# ╠═6cdf0d4c-9c27-47af-8684-e142383727f4
# ╠═21d7edad-6477-46b8-ae82-95796c90be64
# ╟─42fac6d3-5d1c-4608-a9e2-bac922d9d663
# ╠═9027b4e7-fdca-408d-b3db-2fa57cdf91e1
# ╠═90ba2d81-3736-470a-8151-72d52e92826a
# ╟─b89072ad-a499-4837-9325-c9410c25b6e5
# ╟─0c591cc5-3d03-4038-a002-1b447de4b628
# ╟─b42a1b2f-a49d-4f3f-a2e3-715ff3e9cd47
# ╠═f5595113-b9d0-4764-884c-095126ae161e
# ╟─36eb7778-d19a-42ec-bbe8-91fb47cffb00
# ╠═29abe7a9-6c60-4b32-957d-85d701033472
# ╠═84b1da4d-e977-4b70-8e21-ea7df1ad72bd
# ╟─af0accb2-223d-432f-a01d-e8bdd561a763
# ╠═a71303ce-25cb-4bc5-98fa-ed3fd48ae36c
# ╟─a94215d6-90ba-4e4f-b623-a52f7e86ca7c
# ╠═9b1f9abe-4a07-4feb-bea8-1b36d2f2675b
# ╠═76ce6d2e-8d42-4c58-9edd-cec464b4b4a4
# ╟─c8a3eacb-893a-4750-92cb-876bfa3c34f0
# ╠═2b9630d8-c2fb-488c-9d84-18d1e5b7cdc9
# ╠═2a15d869-432a-48fb-9d81-a72c8eb67e6f
# ╟─9ab0965a-181d-4a3e-84d9-746485bc070d
# ╠═07392403-40fd-49cf-9ca6-ed70c6909f6a
# ╠═b67925e3-4020-49d8-9872-e3980eb3ad73
# ╟─8622417c-7642-404a-9405-ef0fab2bf62f
# ╟─19146efc-ded9-4ed7-9a5b-46031d736d27
# ╠═cc8950cc-22be-494c-b55e-9ba19571cfb6
# ╠═b24ab29b-57ab-4eba-b81e-f5095af74c1c
# ╠═68ce341f-c14f-4a80-b8c5-cdcdd795207e
# ╟─b06c0a01-b3ac-48d6-b77a-9c3c630627a0
# ╟─6644d2b7-cd07-4476-a13e-4abb7031bab0
# ╠═c0370e95-65db-47ed-8f22-a205c80ede2d
# ╟─95a128f6-bbf0-49cc-bdfb-c427682a3b03
# ╟─20006671-24b7-4210-877a-9ed401ec350a
# ╠═cfbdfe7a-20e4-4a77-8d3e-33ffe9535213
# ╠═f7c71bce-98e3-47a5-936f-31c367a9208c
# ╠═d7dfd004-69e2-42dd-af39-a7620faa706d
# ╠═4ed254a6-2960-41b3-9d59-947abe27b597
# ╟─3df62f81-3a75-4c8d-b93b-0e002f81eeb6
# ╠═09dc2f34-ed91-4046-8b3c-2a88119aa7a6
# ╟─3dc53acf-9b5d-438e-8882-3dfd0fd3651a
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
