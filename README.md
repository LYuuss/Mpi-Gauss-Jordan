# MPI Gauss-Jordan Solver

Implémentation en C/MPI de l'algorithme de Gauss-Jordan pour résoudre un système linéaire :

```txt
A x = b
```

Le programme travaille sur une matrice augmentée :

```txt
[A | b]
```

et retourne le vecteur solution `x`.

## Objectif du projet

Ce projet a pour but de montrer une implémentation parallèle simple de Gauss-Jordan avec MPI.

Chaque processus MPI reçoit un bloc de lignes de la matrice. À chaque étape :

1. recherche parallèle du meilleur pivot ;
2. pivot partiel pour améliorer la stabilité numérique ;
3. normalisation de la ligne pivot ;
4. diffusion de la ligne pivot à tous les processus ;
5. élimination parallèle sur les lignes locales.

## Structure

```txt
mpi-gauss-jordan/
├── Makefile
├── README.md
├── .gitignore
├── data/
│   └── system3.txt
└── src/
    └── gauss_jordan_mpi.c
```

## Format d'entrée

Le fichier d'entrée doit contenir :

```txt
n
a11 a12 ... a1n b1
a21 a22 ... a2n b2
...
an1 an2 ... ann bn
```

Exemple :

```txt
3
2 1 -1 8
-3 -1 2 -11
-2 1 2 -3
```

Ce système correspond à :

```txt
2x + y - z = 8
-3x - y + 2z = -11
-2x + y + 2z = -3
```

Solution attendue :

```txt
x[0] = 2
x[1] = 3
x[2] = -1
```

## Compilation

```bash
make
```

Ou directement :

```bash
mpicc -O2 -Wall -Wextra -std=c11 -o gauss_jordan_mpi src/gauss_jordan_mpi.c -lm
```

## Exécution

Avec 4 processus :

```bash
mpirun -np 4 ./gauss_jordan_mpi data/system3.txt
```

Ou avec le Makefile :

```bash
make run
```

## Exemple de sortie

```txt
Solution du système :
x[0] = 2.0000000000
x[1] = 3.0000000000
x[2] = -1.0000000000
```

## Détail de l'algorithme

L'algorithme de Gauss-Jordan transforme progressivement la matrice augmentée :

```txt
[A | b]
```

en :

```txt
[I | x]
```

où `I` est la matrice identité et `x` le vecteur solution.

Contrairement à l'élimination de Gauss classique, Gauss-Jordan élimine les coefficients au-dessus et au-dessous du pivot. À la fin, la solution est donc directement lisible dans la dernière colonne.

## Parallélisation MPI

La matrice est distribuée par blocs de lignes.

Par exemple, avec une matrice de taille `n = 8` et `4` processus :

```txt
rank 0 : lignes 0 et 1
rank 1 : lignes 2 et 3
rank 2 : lignes 4 et 5
rank 3 : lignes 6 et 7
```

À chaque itération `k` :

- chaque processus cherche localement le meilleur pivot dans sa partie ;
- `MPI_Allreduce` permet de sélectionner le meilleur pivot global ;
- la ligne pivot est éventuellement échangée avec la ligne courante ;
- le processus propriétaire de la ligne pivot la normalise ;
- la ligne pivot est envoyée aux autres processus avec `MPI_Bcast` ;
- chaque processus élimine le coefficient de la colonne `k` sur ses lignes locales.

## Fonctions MPI utilisées

Le projet utilise principalement :

- `MPI_Init`
- `MPI_Comm_rank`
- `MPI_Comm_size`
- `MPI_Bcast`
- `MPI_Scatterv`
- `MPI_Allreduce`
- `MPI_Sendrecv_replace`
- `MPI_Gatherv`
- `MPI_Finalize`

