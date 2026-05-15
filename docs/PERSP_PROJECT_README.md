# `persp_project_layer`

A CUDA layer implementing perspective projection of 3D Gaussian splats into 2D NDC space, with a fully analytical backward pass. One thread per splat; forward and backward each launch a single kernel.

---

## Forward Pass

### Clip-space transform

Given a combined projection-view matrix $PV \in \mathbb{R}^{4 \times 4}$ (column-major, GLM layout), map a world-space position $\mathbf{p} = (x, y, z)^\top$ to homogeneous clip space:

```math
\mathbf{c} = PV \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix} = \begin{bmatrix} c_x \\ c_y \\ c_z \\ c_w \end{bmatrix}
```

> Splats with $c_w \leq 0$ are behind the camera and are culled - $p_z^{\text{ndc}}$ is set to `FLT_MAX` and the rasterizer skips them.

---

### Perspective divide

Divide by the homogeneous coordinate to obtain normalized device coordinates:

```math
\mathbf{p}^{\text{ndc}} = \frac{1}{c_w}\begin{bmatrix} c_x \\ c_y \\ c_z \end{bmatrix}
```

$p_z^{\text{ndc}}$ is retained for depth sorting only and carries no gradient through the covariance projection.

---

### Jacobian of the perspective map

The covariance projection requires $J = \partial \mathbf{p}^{\text{ndc}}_{xy} / \partial \mathbf{p}$. Applying the quotient rule to $p_x^{\text{ndc}} = c_x / c_w$:

```math
\frac{\partial\, p_x^{\text{ndc}}}{\partial\, p_k} = \frac{\partial c_x/\partial p_k \cdot c_w \;-\; c_x \cdot \partial c_w/\partial p_k}{c_w^2}
```

Since $\partial c_x / \partial p_k = PV_{0k}$ and $\partial c_w / \partial p_k = PV_{3k}$ (using zero-indexed row notation):

```math
\begin{aligned}
J &= \begin{bmatrix}
J_{00} & J_{01} & J_{02} \\
J_{10} & J_{11} & J_{12}
\end{bmatrix} \in \mathbb{R}^{2 \times 3} \\[6pt]
&= \begin{bmatrix}
\partial p_x^{\text{ndc}}/\partial x & \partial p_x^{\text{ndc}}/\partial y & \partial p_x^{\text{ndc}}/\partial z \\
\partial p_y^{\text{ndc}}/\partial x & \partial p_y^{\text{ndc}}/\partial y & \partial p_y^{\text{ndc}}/\partial z
\end{bmatrix} \\[6pt]
&= \frac{1}{c_w^2} \begin{bmatrix}
PV_{00}\,c_w - c_x\,PV_{30} & PV_{01}\,c_w - c_x\,PV_{31} & PV_{02}\,c_w - c_x\,PV_{32} \\
PV_{10}\,c_w - c_y\,PV_{30} & PV_{11}\,c_w - c_y\,PV_{31} & PV_{12}\,c_w - c_y\,PV_{32}
\end{bmatrix}
\end{aligned}
```

$J$ is $2 \times 3$ - the third row $\partial p_z^{\text{ndc}} / \partial \mathbf{p}$ is omitted because depth receives no gradient.

---

### Covariance projection

A Gaussian with covariance $\Sigma_{3D}$ transforms under the linear map $J$ as:

```math
\Sigma_{2D} = J\,\Sigma_{3D}\,J^\top
```

$\Sigma_{3D} \in \mathbb{R}^{3 \times 3}$ is symmetric; only the upper triangle is stored:

```math
\Sigma_{3D} = \begin{bmatrix} \sigma_{xx} & \sigma_{xy} & \sigma_{xz} \\ \sigma_{xy} & \sigma_{yy} & \sigma_{yz} \\ \sigma_{xz} & \sigma_{yz} & \sigma_{zz} \end{bmatrix}
```

$\Sigma_{2D} \in \mathbb{R}^{2 \times 2}$ is also symmetric; only the upper triangle is stored:

```math
\Sigma_{2D} = \begin{bmatrix} \sigma_{xx}^{2D} & \sigma_{xy}^{2D} \\ \sigma_{xy}^{2D} & \sigma_{yy}^{2D} \end{bmatrix}
```

giving three entries, where:

```math
\sigma_{xx}^{2D} = J_{00}^2\,\sigma_{xx} + J_{01}^2\,\sigma_{yy} + J_{02}^2\,\sigma_{zz} + 2J_{00}J_{01}\,\sigma_{xy} + 2J_{00}J_{02}\,\sigma_{xz} + 2J_{01}J_{02}\,\sigma_{yz}
```

```math
\sigma_{xy}^{2D} = J_{00}J_{10}\,\sigma_{xx} + J_{01}J_{11}\,\sigma_{yy} + J_{02}J_{12}\,\sigma_{zz} + (J_{00}J_{11}+J_{01}J_{10})\,\sigma_{xy} + (J_{00}J_{12}+J_{02}J_{10})\,\sigma_{xz} + (J_{01}J_{12}+J_{02}J_{11})\,\sigma_{yz}
```

```math
\sigma_{yy}^{2D} = J_{10}^2\,\sigma_{xx} + J_{11}^2\,\sigma_{yy} + J_{12}^2\,\sigma_{zz} + 2J_{10}J_{11}\,\sigma_{xy} + 2J_{10}J_{12}\,\sigma_{xz} + 2J_{11}J_{12}\,\sigma_{yz}
```

---

## Backward Pass

The Jacobian $J$ is recomputed from the stored world positions rather than read from a saved buffer - cheaper in memory traffic than caching 6 floats per splat.

---

### Position gradient

The NDC map $\mathbf{p}^{\text{ndc}}_{xy} = J\,\mathbf{p}$ is linear in $\mathbf{p}$ with Jacobian $J$, so by the chain rule:

```math
\frac{\partial L}{\partial \mathbf{p}} = J^\top \frac{\partial L}{\partial \mathbf{p}^{\text{ndc}}_{xy}}
```

Explicitly, letting $g_x = \partial L / \partial p_x^{\text{ndc}}$ and $g_y = \partial L / \partial p_y^{\text{ndc}}$:

```math
\begin{bmatrix} \partial L/\partial x \\ \partial L/\partial y \\ \partial L/\partial z \end{bmatrix} = \begin{bmatrix} J_{00}\,g_x + J_{10}\,g_y \\ J_{01}\,g_x + J_{11}\,g_y \\ J_{02}\,g_x + J_{12}\,g_y \end{bmatrix}
```

---

### Covariance gradient

For the map $\Sigma_{2D} = J\,\Sigma_{3D}\,J^\top$, differentiating through both $J$ factors gives:

```math
\frac{\partial L}{\partial \Sigma_{3D}} = J^\top H\, J = \begin{bmatrix}
\dfrac{\partial L}{\partial \sigma_{xx}} & \dfrac{\partial L}{\partial \sigma_{xy}} & \dfrac{\partial L}{\partial \sigma_{xz}} \\[10pt]
\dfrac{\partial L}{\partial \sigma_{xy}} & \dfrac{\partial L}{\partial \sigma_{yy}} & \dfrac{\partial L}{\partial \sigma_{yz}} \\[10pt]
\dfrac{\partial L}{\partial \sigma_{xz}} & \dfrac{\partial L}{\partial \sigma_{yz}} & \dfrac{\partial L}{\partial \sigma_{zz}}
\end{bmatrix}
```

Note that:

```math
\frac{\partial L}{\partial \Sigma_{2D}} = \begin{bmatrix} \dfrac{\partial L}{\partial\sigma_{xx}^{2D}} & \dfrac{\partial L}{\partial\sigma_{xy}^{2D}} \\ \dfrac{\partial L}{\partial\sigma_{xy}^{2D}} & \dfrac{\partial L}{\partial\sigma_{yy}^{2D}} \end{bmatrix}
```

letting $g_{xx} = \partial L/\partial\sigma_{xx}^{2D}$, $g_{xy} = \partial L/\partial\sigma_{xy}^{2D}$, $g_{yy} = \partial L/\partial\sigma_{yy}^{2D}$, $H \in \mathbb{R}^{2\times2}$ is:

```math
H = \begin{bmatrix} g_{xx} & h \\ h & g_{yy} \end{bmatrix}, \qquad h = \tfrac{1}{2}\,g_{xy}
```

The factor of $\tfrac{1}{2}$ on the off-diagonal arises from symmetry: $\Sigma_{3D}$ stores each off-diagonal element once, but the full-matrix gradient accumulates from both the $[i,j]$ and $[j,i]$ positions. This halving is consistent with the `GaussActivLayer` backward convention $\partial L/\partial M = 2\,G_{\text{full}}\,M$.

Expanding $J^\top H J$ into the six unique entries of the symmetric output:

```math
\frac{\partial L}{\partial \sigma_{xx}} = J_{00}^2\,g_{xx} + J_{00}J_{10}\,g_{xy} + J_{10}^2\,g_{yy}
```

```math
\frac{\partial L}{\partial \sigma_{xy}} = J_{00}J_{01}\,g_{xx} + h\!\left(J_{00}J_{11} + J_{10}J_{01}\right) + J_{10}J_{11}\,g_{yy}
```

```math
\frac{\partial L}{\partial \sigma_{xz}} = J_{00}J_{02}\,g_{xx} + h\!\left(J_{00}J_{12} + J_{10}J_{02}\right) + J_{10}J_{12}\,g_{yy}
```

```math
\frac{\partial L}{\partial \sigma_{yy}} = J_{01}^2\,g_{xx} + J_{01}J_{11}\,g_{xy} + J_{11}^2\,g_{yy}
```

```math
\frac{\partial L}{\partial \sigma_{yz}} = J_{01}J_{02}\,g_{xx} + h\!\left(J_{01}J_{12} + J_{11}J_{02}\right) + J_{11}J_{12}\,g_{yy}
```

```math
\frac{\partial L}{\partial \sigma_{zz}} = J_{02}^2\,g_{xx} + J_{02}J_{12}\,g_{xy} + J_{12}^2\,g_{yy}
```

---

## Notes

- $PV$ is stored in `__constant__` memory and treated as a constant - no gradient flows to the camera.
- Color and opacity pass through both kernels unchanged.
- Culled splats ($c_w \leq 0$) receive zero on all gradient outputs in the backward pass.
