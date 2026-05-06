# `gauss_activ_layer`

A CUDA layer that maps raw Gaussian parameters to the quantities consumed by `persp_project_layer`: 3D covariance, linear-space color, and opacity. One thread per splat; forward and backward each launch a single kernel.

---

## Forward Pass

### Quaternion normalization

The raw stored quaternion $\mathbf{q}_{\text{raw}} = (q_w, q_x, q_y, q_z)^\top$ may not be constrained to unit length, so it is normalized before use:

```math
\hat{\mathbf{q}} = \frac{\mathbf{q}_{\text{raw}}}{\|\mathbf{q}_{\text{raw}}\|}
```

The normalized quaternion $\hat{\mathbf{q}} = (\hat{q}_w, \hat{q}_x, \hat{q}_y, \hat{q}_z)^\top$ is then converted to a rotation matrix $R \in \mathbb{R}^{3 \times 3}$.

---

### Scale activation

Log-scale inputs are exponentiated to ensure positivity:

```math
s_x = e^{\tilde{s}_x}, \qquad s_y = e^{\tilde{s}_y}, \qquad s_z = e^{\tilde{s}_z}
```

---

### Covariance factor M

The covariance factor $M \in \mathbb{R}^{3 \times 3}$ combines rotation and scale:

```math
M = R \, S, \qquad S = \begin{bmatrix} s_x & 0 & 0 \\ 0 & s_y & 0 \\ 0 & 0 & s_z \end{bmatrix}
```

Column-wise this is just $M_{:,k} = s_k \, R_{:,k}$, so no full matrix multiply is needed.

---

### 3D covariance

```math
\Sigma_{3D} = M M^\top = \begin{bmatrix} \sigma_{xx} & \sigma_{xy} & \sigma_{xz} \\ \sigma_{xy} & \sigma_{yy} & \sigma_{yz} \\ \sigma_{xz} & \sigma_{yz} & \sigma_{zz} \end{bmatrix}
```

$\Sigma_{3D}$ is symmetric; only the upper triangle is stored ($\sigma_{xx}, \sigma_{xy}, \sigma_{xz}, \sigma_{yy}, \sigma_{yz}, \sigma_{zz}$).

---

### SH color

The view direction from the splat to the camera is computed and normalized:

```math
\mathbf{v} = \frac{\mathbf{p}_{\text{cam}} - \mathbf{p}_{\text{splat}}}{\|\mathbf{p}_{\text{cam}} - \mathbf{p}_{\text{splat}}\|}
```

The linear-space color is evaluated by summing spherical harmonic basis functions up to the active degree $\ell_{\max} \in \{0,1,2,3\}$, then shifting and clamping:

```math
c = \text{clamp}(\sum_{\ell=0}^{\ell_{\max}} \sum_{m=-\ell}^{\ell} f_{\ell m} \, Y_{\ell m}(\mathbf{v}) + 0.5,\, 0,\, 1)
```

The coefficients $f_{\ell m} \in \mathbb{R}^3$ are stored per splat and hold separate values for R, G, B; the formula is applied independently per channel.

> View direction is treated as a constant in the backward pass (no gradient flows to splat position through color). This is the standard 3DGS approximation.

---

### Sigmoid opacity

```math
\alpha = \sigma(\tilde{\alpha}) = \frac{1}{1 + e^{-\tilde{\alpha}}}
```

---

## Backward Pass

The backward kernel recomputes $R$, $s_{x/y/z}$, $M$, and $\mathbf{v}$ from the raw inputs rather than caching them - these are pure intermediates that only live inside the forward pass. The forward color $c$ and opacity $\alpha$ are read from the saved layer outputs, since both are needed as gates ($c$ for the clamp, $\alpha$ for the sigmoid derivative).

---

### Covariance gradient: dL/dM

Upstream delivers $G_{\text{full}}$ - the gradient of $L$ w.r.t. $\Sigma_{3D}$ in the **full-matrix convention** (off-diagonals are already halved by `persp_project_layer`):

```math
G_{\text{full}} = \begin{bmatrix} g_{xx} & g_{xy} & g_{xz} \\ g_{xy} & g_{yy} & g_{yz} \\ g_{xz} & g_{yz} & g_{zz} \end{bmatrix} = G_{\text{full}}^\top
```

Because $\Sigma_{3D} = M M^\top$ and $G_{\text{full}}$ is symmetric, the chain rule gives:

```math
\frac{\partial L}{\partial M} = 2\,G_{\text{full}}\,M
```

Expanding all nine entries (using $m_{pq} = M[p,q]$):

```math
\begin{aligned}
\frac{\partial L}{\partial m_{00}} &= 2(g_{xx}\,m_{00} + g_{xy}\,m_{10} + g_{xz}\,m_{20}) \\
\frac{\partial L}{\partial m_{01}} &= 2(g_{xx}\,m_{01} + g_{xy}\,m_{11} + g_{xz}\,m_{21}) \\
\frac{\partial L}{\partial m_{02}} &= 2(g_{xx}\,m_{02} + g_{xy}\,m_{12} + g_{xz}\,m_{22}) \\[16pt]
\frac{\partial L}{\partial m_{10}} &= 2(g_{xy}\,m_{00} + g_{yy}\,m_{10} + g_{yz}\,m_{20}) \\
\frac{\partial L}{\partial m_{11}} &= 2(g_{xy}\,m_{01} + g_{yy}\,m_{11} + g_{yz}\,m_{21}) \\
\frac{\partial L}{\partial m_{12}} &= 2(g_{xy}\,m_{02} + g_{yy}\,m_{12} + g_{yz}\,m_{22}) \\[16pt]
\frac{\partial L}{\partial m_{20}} &= 2(g_{xz}\,m_{00} + g_{yz}\,m_{10} + g_{zz}\,m_{20}) \\
\frac{\partial L}{\partial m_{21}} &= 2(g_{xz}\,m_{01} + g_{yz}\,m_{11} + g_{zz}\,m_{21}) \\
\frac{\partial L}{\partial m_{22}} &= 2(g_{xz}\,m_{02} + g_{yz}\,m_{12} + g_{zz}\,m_{22})
\end{aligned}
```

---

### Scale gradient

Since $M_{:,k} = s_k\,R_{:,k}$, the scale gradients are dot products of the corresponding columns of $\mathbf{dM}$ and $R$ (using $r_{pq} = R[p,q]$):

```math
\begin{aligned}
\frac{\partial L}{\partial s_x} &= \frac{\partial L}{\partial m_{00}}\,r_{00} + \frac{\partial L}{\partial m_{10}}\,r_{10} + \frac{\partial L}{\partial m_{20}}\,r_{20} \\
\frac{\partial L}{\partial s_y} &= \frac{\partial L}{\partial m_{01}}\,r_{01} + \frac{\partial L}{\partial m_{11}}\,r_{11} + \frac{\partial L}{\partial m_{21}}\,r_{21} \\
\frac{\partial L}{\partial s_z} &= \frac{\partial L}{\partial m_{02}}\,r_{02} + \frac{\partial L}{\partial m_{12}}\,r_{12} + \frac{\partial L}{\partial m_{22}}\,r_{22}
\end{aligned}
```

Chaining through the log-scale activation $s_k = e^{\tilde{s}_k}$:

```math
\begin{aligned}
\frac{\partial L}{\partial \tilde{s}_x} &= \frac{\partial L}{\partial s_x} \cdot s_x \\
\frac{\partial L}{\partial \tilde{s}_y} &= \frac{\partial L}{\partial s_y} \cdot s_y \\
\frac{\partial L}{\partial \tilde{s}_z} &= \frac{\partial L}{\partial s_z} \cdot s_z
\end{aligned}
```

---

### Rotation gradient

Since $M = R\,S$, differentiating w.r.t. $R$ with $S$ fixed:

```math
\frac{\partial L}{\partial R} = \frac{\partial L}{\partial M} \cdot S
```

Column-wise:

```math
\frac{\partial L}{\partial R_{:,k}} = s_k\,\frac{\partial L}{\partial M_{:,k}}
```

---

### Quaternion gradient

The rotation matrix $R$ is a function of the normalized quaternion $\hat{\mathbf{q}}$. Differentiating the standard quaternion-to-matrix formula gives $\partial L / \partial \hat{\mathbf{q}}$.

Since $\hat{\mathbf{q}}$ is a unit-length vector, the Jacobian of the normalization map is:

```math
\frac{\partial \hat{\mathbf{q}}}{\partial \mathbf{q}_{\text{raw}}} = \frac{1}{\|\mathbf{q}_{\text{raw}}\|}\left(I - \hat{\mathbf{q}}\,\hat{\mathbf{q}}^\top\right)
```

Applying the chain rule:

```math
\frac{\partial L}{\partial \mathbf{q}_{\text{raw}}} = \frac{\partial L}{\partial \hat{\mathbf{q}}} \cdot \frac{\partial \hat{\mathbf{q}}}{\partial \mathbf{q}_{\text{raw}}} = \frac{1}{\|\mathbf{q}_{\text{raw}}\|}\left(\frac{\partial L}{\partial \hat{\mathbf{q}}} - \left(\frac{\partial L}{\partial \hat{\mathbf{q}}} \cdot \hat{\mathbf{q}}\right)\hat{\mathbf{q}}\right)
```

The dot product term removes the component of the gradient along $\hat{\mathbf{q}}$ itself, which has no effect on the normalized output.

---

### SH gradient

The clamp acts as a gate: its gradient is 1 in the interior and 0 at a boundary. Including that gate explicitly:

```math
\frac{\partial L}{\partial f_{\ell m}} = \frac{\partial L}{\partial c} \cdot \mathbf{1}[0 < c < 1] \cdot Y_{\ell m}(\mathbf{v})
```

Applied independently per color channel.

---

### Sigmoid gradient

```math
\frac{\partial L}{\partial \tilde{\alpha}} = \frac{\partial L}{\partial \alpha} \cdot \alpha\,(1 - \alpha)
```
