import torch
import time
import math
import torch.nn.functional as F
import lltm_cpp


class LLTMFunction_CUDA(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        weights: torch.Tensor,
        bias: torch.Tensor,
        old_h: torch.Tensor,
        old_cell: torch.Tensor,
    ):
        outputs = lltm_cpp.forward_cuda(input, weights, bias, old_h, old_cell)
        new_h: torch.Tensor
        new_cell: torch.Tensor
        new_h, new_cell = outputs[:2]
        variables = outputs[1:] + [weights]
        ctx.save_for_backward(*variables)

        return new_h, new_cell

    @staticmethod
    def backward(  # type: ignore
        ctx: torch.autograd.function.FunctionCtx,
        grad_h: torch.Tensor,
        grad_cell: torch.Tensor,
    ):
        outputs = lltm_cpp.backward_cuda(
            grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_tensors  # type: ignore
        )
        d_old_h, d_input, d_weights, d_bias, d_old_cell = outputs
        return d_input, d_weights, d_bias, d_old_h, d_old_cell


class LLTMFunction_CPU(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        weights: torch.Tensor,
        bias: torch.Tensor,
        old_h: torch.Tensor,
        old_cell: torch.Tensor,
    ):
        outputs = lltm_cpp.forward_cpu(input, weights, bias, old_h, old_cell)
        new_h: torch.Tensor
        new_cell: torch.Tensor
        new_h, new_cell = outputs[:2]
        variables = outputs[1:] + [weights]
        ctx.save_for_backward(*variables)

        return new_h, new_cell

    @staticmethod
    def backward(  # type: ignore
        ctx: torch.autograd.function.FunctionCtx,
        grad_h: torch.Tensor,
        grad_cell: torch.Tensor,
    ):
        outputs = lltm_cpp.backward_cpu(
            grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_tensors  # type: ignore
        )
        d_old_h, d_input, d_weights, d_bias, d_old_cell = outputs
        return d_input, d_weights, d_bias, d_old_h, d_old_cell


class LLTM(torch.nn.Module):
    def __init__(self, input_features: int, state_size: int, version: str = ""):
        super(LLTM, self).__init__()  # type: ignore
        self.input_features = input_features
        self.state_size = state_size
        # 3 * state_size for input gate, output gate and candidate cell gate.
        # input_features + state_size because we will multiply with [input, h].
        self.weights = torch.nn.Parameter(
            torch.empty(3 * state_size, input_features + state_size)
        )
        self.bias = torch.nn.Parameter(torch.empty(3 * state_size))
        self.version = version
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(
        self, input: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        match self.version:
            case "cpp":
                return LLTMFunction_CPU.apply(input, self.weights, self.bias, *state)  # type: ignore
            case "cuda":
                return LLTMFunction_CUDA.apply(input, self.weights, self.bias, *state)  # type: ignore
            case _:
                old_h, old_cell = state  # b * s, b * s
                X = torch.cat([old_h, input], dim=1)  # b * (s + i)

                # Compute the input, output and candidate cell gates with one MM.
                gate_weights = F.linear(X, self.weights, self.bias)
                # gate_weights: b * 3s
                # Split the combined gate weight matrix into its components.
                gates = gate_weights.chunk(3, dim=1)

                input_gate = torch.sigmoid(gates[0])
                output_gate = torch.sigmoid(gates[1])
                # Here we use an ELU instead of the usual tanh.
                candidate_cell = F.elu(gates[2])

                # Compute the new cell state.
                new_cell = old_cell + candidate_cell * input_gate
                # Compute the new hidden state and output.
                new_h = torch.tanh(new_cell) * output_gate

                return new_h, new_cell


batch_size = 16
input_features = 32
state_size = 128

X0 = torch.randn(batch_size, input_features)
h0 = torch.randn(batch_size, state_size)
C0 = torch.randn(batch_size, state_size)

assert torch.cuda.is_available()

for i in range(2):
    for device in ["cuda", "cpu"]:
        X = X0.to(device=device)
        h = h0.to(device=device)
        C = C0.to(device=device)
        for version in ["pytorch", "cpp", "cuda"]:
            if version == "cuda" and device == "cpu":
                continue
            rnn = LLTM(input_features, state_size, version=version).to(
                device=device
            )

            forward = 0
            backward = 0
            for _ in range(1000):
                start = time.time()
                new_h, new_C = rnn(X, (h, C))
                if device == "cuda":
                    torch.cuda.synchronize()
                forward += time.time() - start

                start = time.time()
                (new_h.sum() + new_C.sum()).backward()
                if device == "cuda":
                    torch.cuda.synchronize()
                backward += time.time() - start

            print(
                "Device: {}, Version: {}, Forward: {:.3f} s | Backward {:.3f} s".format(
                    device, version, forward, backward
                )
            )
