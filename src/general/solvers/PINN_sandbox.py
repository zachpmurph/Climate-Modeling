import torch
from torch import nn


# PINNs often benefit from float64 numerical precision.
torch.set_default_dtype(torch.float64)
torch.manual_seed(1)

def make_collocation_points(x, y, t):
    X, Y, T = torch.meshgrid(
        x,
        y,
        t,
        indexing="ij",
    )

    coordinates = torch.stack(
        [
            X.flatten(),
            Y.flatten(),
            T.flatten(),
        ],
        dim=1,
    )

    # Required because we need derivatives with respect to x, y and t.
    coordinates.requires_grad_(True)

    return coordinates

def make_initial_points(x, y):
    t = torch.zeros_like(x)
    initial_coords = make_collocation_points(x, y, t)

    return initial_coords

def make_targets(number_initial):
    h = torch.ones(number_initial, 1)
    u = torch.zeros(number_initial, 1)
    v = torch.zeros(number_initial, 1)
    return h, u, v

def coordinate_gradient(field, coordinates):
    """
    Return derivatives of `field` with respect to x, y and t.

    field:       shape (N, 1)
    coordinates: shape (N, 3)
    """
    gradient = torch.autograd.grad(
        outputs=field,
        inputs=coordinates,
        grad_outputs=torch.ones_like(field),
        create_graph=True,
    )[0]

    derivative_x = gradient[:, 0:1]
    derivative_y = gradient[:, 1:2]
    derivative_t = gradient[:, 2:3]

    return derivative_x, derivative_y, derivative_t

def continuity_residual(model, coordinates):
    h, u, v = model(coordinates)

    discharge_x = h * u
    discharge_y = h * v

    _, _, dh_dt = coordinate_gradient(h, coordinates)

    dhu_dx, _, _ = coordinate_gradient(
        discharge_x,
        coordinates,
    )

    _, dhv_dy, _ = coordinate_gradient(
        discharge_y,
        coordinates,
    )

    residual = dh_dt + dhu_dx + dhv_dy

    return residual

def physics_loss(model, coordinates):
    residual = continuity_residual(
        model,
        coordinates,
    )

    return torch.mean(residual**2)

def initial_loss(model, initial_coords, target_h, target_u, target_v):
    pred_h, pred_u, pred_v = model(initial_coords)
    loss_h = torch.mean((pred_h - target_h)**2)
    loss_u = torch.mean((pred_u - target_u)**2)
    loss_v = torch.mean((pred_v - target_v)**2)

    return loss_h + loss_u + loss_v

def total_loss(model, coordinates, initial_coords, target_h, target_u, target_v, initial_weight = 1.0):
    L_p = physics_loss(model, coordinates)
    L_i = initial_loss(model, initial_coords, target_h, target_u, target_v)

    return L_p + initial_weight * L_i, L_p, L_i * initial_weight

class PINN(nn.Module):
    """Map coordinates (x, y, t) to water state (h, u, v)."""

    def __init__(self, hidden_width=128):
        super().__init__()

        self.hidden_layer = nn.Linear(
            in_features=3,
            out_features=hidden_width,
        )

        self.activation = nn.SiLU()

        self.output_layer = nn.Linear(
            in_features=hidden_width,
            out_features=3,
        )

        self.positive_depth = nn.Softplus()

    def forward(self, coordinates):
        """
        coordinates: shape (N, 3)

        Column 0: x
        Column 1: y
        Column 2: t
        """
        hidden = self.hidden_layer(coordinates)
        hidden = self.activation(hidden)
        raw_output = self.output_layer(hidden)

        raw_h = raw_output[:, 0:1]
        u = raw_output[:, 1:2]
        v = raw_output[:, 2:3]

        # Enforce h > 0 while preserving differentiability.
        h = self.positive_depth(raw_h)

        return h, u, v

def main():
    model = PINN()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    x = torch.linspace(0.0, 10.0, 10)
    y = torch.linspace(0.0, 10.0, 10)
    t = torch.linspace(0.0, 10.0, 10)
    number_initial = len(x) * len(y) * len(t)

    coordinates = make_collocation_points(x, y, t)

    h, u, v = model(coordinates)
    initial_coords = make_initial_points(x, y)
    print(initial_coords.shape)
    target_h, target_u, target_v = make_targets(number_initial)

    num_epoch = 1000

    for epoch in range(num_epoch):

        #create new differentiable coords
        interior_coordinates = (
            coordinates
            .detach()
            .clone()
            .requires_grad_(True)
        )

        #reset grads
        optimizer.zero_grad(set_to_none = True)

        #calculate loss
        loss, loss_p, loss_i = total_loss(model, interior_coordinates, initial_coords, target_h, target_u, target_v)

        #backprop loss
        loss.backward()

        #step optimizer
        optimizer.step()

        if epoch % 100 == 0:
            print(
                f"epoch={epoch:5d} "
                f"total={loss.item():.6e} "
                f"physics={loss_p.item():.6e} "
                f"initial={loss_i.item():.6e}"
            )

    print("Coordinate shape:", coordinates.shape)
    print("Depth shape:", h.shape)
    print("Velocity-x shape:", u.shape)
    print("Velocity-y shape:", v.shape)
    print("Initial total loss:", loss.item())
    print("Initial physics loss:", loss_p.item())
    print("Initial conditions loss:", loss_i.item())

if __name__ == "__main__":
    main()