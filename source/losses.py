import torch
import torch.nn.functional as F


def custom_binary_cross_entropy_with_logits(inputs_list, targets, weights, reduction_after_reweighting='sum'):
    """
    :param inputs_list: List of tensors of predictions or None-s. Each tensor should be of shape (batch_size, num_classes).
    :param targets: Tensor of true labels of shape (batch_size, num_classes)
    :param weights: Tensor of weights of shape (batch_size, num_classes)
    :return: scalar loss
    """
    if reduction_after_reweighting != 'sum':
        raise NotImplementedError("Only reduction_after_reweighting='sum' is implemented.")

    total_loss = 0
    total_valid_inputs = 0
    for inputs in inputs_list:
        if inputs is not None:
            assert inputs.shape == targets.shape == weights.shape
            loss_tensor = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            loss_tensor = loss_tensor * weights
            # .sum() returns the sum of all elements in the input tensor: https://pytorch.org/docs/stable/generated/torch.sum.html
            total_loss += loss_tensor.sum()
            total_valid_inputs += 1

    total_weights_sum = weights.sum()
    # Check if total_weights_sum is zero - gradient calculated using this loss will be zero
    if total_weights_sum.item() == 0:
        return torch.tensor(0.0, device=weights.device, dtype=weights.dtype, requires_grad=True)

    return total_loss / (total_valid_inputs * total_weights_sum)
