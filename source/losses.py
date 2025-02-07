import torch
import torch.nn.functional as F


def custom_cross_entropy_loss(inputs_list, targets, weights, loss_type='binary_cross_entropy', reduction_after_reweighting='sum'):
    """
    :param inputs_list: List of tensors of unnormalised predictions (logits) or None-s. Each tensor should be of shape (batch_size, num_classes).
    :param targets: Tensor of true labels of shape (batch_size, num_classes)
    :param weights: Tensor of weights of shape (batch_size, num_classes)
    :param loss_type: Type of loss to compute ('binary_cross_entropy' or 'cross_entropy')
    :param reduction_after_reweighting: Reduction method after reweighting ('sum' is implemented)
    :return: scalar loss
    """
    if reduction_after_reweighting != 'sum':
        raise NotImplementedError("Only reduction_after_reweighting='sum' is implemented.")

    total_weights_sum = weights.sum()
    # Check if total_weights_sum is zero - gradient calculated using this loss will be zero
    if total_weights_sum.item() == 0:
        return torch.tensor(0.0, device=weights.device, dtype=weights.dtype, requires_grad=True)

    if loss_type == 'cross_entropy':
        assert torch.all(weights > 0), "All weights must be greater than 0 for cross_entropy loss."

    total_loss = 0
    total_valid_inputs = 0
    for inputs in inputs_list:
        if inputs is not None:
            assert inputs.shape == targets.shape == weights.shape
            if loss_type == 'binary_cross_entropy':
                loss_tensor = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none') # (batch_size, num_classes)
                loss_tensor = loss_tensor * weights # elementwise multiplication (batch_size, num_classes)
            elif loss_type == 'cross_entropy':
                loss_tensor = F.cross_entropy(inputs, targets, reduction='none') # (batch_size,)
                loss_tensor = loss_tensor * weights.sum(dim=1) # elementwise multiplication (batch_size,)
            else:
                raise ValueError("Invalid loss_type. Choose 'binary_cross_entropy' or 'cross_entropy'.")
            
            # .sum() returns the sum of all elements in the input tensor: https://pytorch.org/docs/stable/generated/torch.sum.html
            total_loss += loss_tensor.sum()
            total_valid_inputs += 1

    return total_loss / (total_valid_inputs * total_weights_sum)