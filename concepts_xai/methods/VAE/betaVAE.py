from methods.VAE.baseVAE import BaseVAE

class BetaVAE(BaseVAE):
  """BetaVAE model."""

  def __init__(self, latent_dim, encoder_fn, decoder_fn, loss_fn, input_shape, beta=1,  **kwargs):
    """Creates a beta-VAE model.

    Implementing Eq. 4 of "beta-VAE: Learning Basic Visual Concepts with a
    Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl).

    :param beta: Hyperparameter for the regularizer.
    """
    super(BetaVAE, self).__init__(latent_dim, encoder_fn, decoder_fn, loss_fn, input_shape, **kwargs)
    self.beta = beta

  def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
    del z_mean, z_logvar, z_sampled
    return self.beta * kl_loss





