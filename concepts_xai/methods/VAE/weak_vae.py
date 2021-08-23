import tensorflow as tf
from concepts_xai.methods.VAE.baseVAE import BaseVAE, compute_gaussian_kl


class GroupVAEBase(BaseVAE):
    """Beta-VAE with averaging from https://arxiv.org/abs/1809.02383."""

    def __init__(
        self,
        encoder,
        decoder,
        loss_fn,
        beta=1,
        **kwargs,
    ):
        """
        Creates a beta-VAE model with additional averaging for weak
        supervision.
        Based on https://arxiv.org/abs/1809.02383.

        :param beta: Hyperparameter for KL divergence.
        """
        super(GroupVAEBase, self).__init__(
            encoder=encoder,
            decoder=decoder,
            loss_fn=loss_fn,
            **kwargs,
        )
        self.beta = beta
        self.metric_names.append("regularizer")
        self.metrics_dict["regularizer"] = tf.keras.metrics.Mean(
            name="regularizer"
        )

    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        del z_mean, z_logvar, z_sampled
        return self.beta * kl_loss

    def _split_sample_pairs(self, x):
        '''
        Note: each point contains two frames stacked along the first dim and
        integer labels.
        '''
        if isinstance(x, tuple):
            assert len(x) == 2, f'Expected samples to come as pairs f{x}'
            return (x[0], x[1])
        data_shape = x.get_shape().as_list()[1:]
        assert data_shape[0] % 2 == 0, (
            "1st dimension of concatenated pairs assumed to be even"
        )
        data_shape[0] = data_shape[0] // 2
        return x[:, :data_shape[0], ...], x[:, data_shape[0]:, ...]

    def _compute_losses_weak(self, x_1, x_2, is_training, labels=None):

        z_mean, z_logvar = self.encoder(x_1, training=is_training)
        z_mean_2, z_logvar_2 = self.encoder(x_2, training=is_training)
        if labels is not None:
            labels = tf.squeeze(
                tf.one_hot(labels, z_mean.get_shape().as_list()[1])
            )
        kl_per_point = compute_kl(z_mean, z_mean_2, z_logvar, z_logvar_2)

        new_mean = 0.5 * z_mean + 0.5 * z_mean_2
        var_1 = tf.exp(z_logvar)
        var_2 = tf.exp(z_logvar_2)
        new_log_var = tf.math.log(0.5*var_1 + 0.5*var_2)

        mean_sample_1, log_var_sample_1 = self.aggregate(
            z_mean,
            z_logvar,
            new_mean,
            new_log_var,
            labels,
            kl_per_point,
        )
        mean_sample_2, log_var_sample_2 = self.aggregate(
            z_mean_2,
            z_logvar_2,
            new_mean,
            new_log_var,
            labels,
            kl_per_point,
        )

        z_sampled_1 = self.sample_from_latent_distribution(
            mean_sample_1,
            log_var_sample_1,
        )
        z_sampled_2 = self.sample_from_latent_distribution(
            mean_sample_2,
            log_var_sample_2,
        )

        reconstructions_1 = self.decoder(
            z_sampled_1,
            training=is_training
        )
        reconstructions_2 = self.decoder(
            z_sampled_2,
            training=is_training,
        )

        per_sample_loss_1 = self.loss_fn(x_1, reconstructions_1)
        per_sample_loss_2 = self.loss_fn(x_2, reconstructions_2)
        reconstruction_loss_1 = tf.reduce_mean(per_sample_loss_1)
        reconstruction_loss_2 = tf.reduce_mean(per_sample_loss_2)
        reconstruction_loss = (
            0.5 * reconstruction_loss_1 + 0.5 * reconstruction_loss_2
        )

        kl_loss_1 = compute_gaussian_kl(mean_sample_1, log_var_sample_1)
        kl_loss_2 = compute_gaussian_kl(mean_sample_2, log_var_sample_2)
        kl_loss = 0.5 * kl_loss_1 + 0.5 * kl_loss_2

        regularizer = self.regularizer(kl_loss, None, None, None)

        loss = tf.add(reconstruction_loss, regularizer, name="loss")
        elbo = tf.add(reconstruction_loss, kl_loss, name="elbo")

        return reconstruction_loss, regularizer, loss, elbo

    def _split_labels(self, inputs):
        return inputs, None

    def train_step(self, inputs):
        x, labels = self._split_labels(inputs)
        x_1, x_2 = self._split_sample_pairs(x)

        with tf.GradientTape() as tape:
            rec_loss, regularizer, loss, elbo = self._compute_losses_weak(
                x_1=x_1,
                x_2=x_2,
                is_training=True,
                labels=labels,
            )

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )
        self.update_metrics([
            ("loss", loss),
            ("reconstruction_loss", rec_loss),
            ("elbo", elbo),
            ("regularizer",regularizer),
        ])

        return {
            name: self.metrics_dict[name].result()
            for name in self.metric_names
        }

    def test_step(self, inputs):
        x, labels = self._split_labels(inputs)
        x_1, x_2 = self._split_sample_pairs(x)
        rec_loss, regularizer, loss, elbo = self._compute_losses_weak(
            x_1=x_1,
            x_2=x_2,
            is_training=False,
            labels=labels,
        )
        self.update_metrics([
            ("loss", loss),
            ("reconstruction_loss", rec_loss),
            ("elbo", elbo),
            ("regularizer", regularizer)
        ])

        return {name: self.metrics_dict[name].result() for name in self.metric_names}


class GroupVAEArgmax(GroupVAEBase):
    """Class implementing the group-VAE without any label."""

    def aggregate(
        self,
        z_mean,
        z_logvar,
        new_mean,
        new_log_var,
        labels,
        kl_per_point,
    ):
        return aggregate_argmax(
            z_mean,
            z_logvar,
            new_mean,
            new_log_var,
            kl_per_point,
        )


class GroupVAELabels(GroupVAEBase):
    """Class implementing the group-VAE with labels on which factor is shared."""

    def _split_labels(self, inputs):
        return inputs

    def aggregate(
        self,
        z_mean,
        z_logvar,
        new_mean,
        new_log_var,
        labels,
        kl_per_point,
    ):
        return aggregate_labels(
            z_mean,
            z_logvar,
            new_mean,
            new_log_var,
            labels,
            kl_per_point,
        )


class MLVae(GroupVAEBase):
    """Beta-VAE with averaging from https://arxiv.org/abs/1705.08841."""

    def _compute_losses_weak(self, x_1, x_2, is_training, labels=None):
        z_mean, z_logvar = self.encoder(x_1, training=is_training)
        z_mean_2, z_logvar_2 = self.encoder(x_2, training=is_training)
        if labels is not None:
            labels = tf.squeeze(
                tf.one_hot(labels, z_mean.get_shape().as_list()[1])
            )
        kl_per_point = compute_kl(z_mean, z_mean_2, z_logvar, z_logvar_2)

        var_1 = tf.exp(z_logvar)
        var_2 = tf.exp(z_logvar_2)
        new_var = 2 * var_1 * var_2 / (var_1 + var_2)
        new_mean = ((z_mean / var_1) + (z_mean_2 / var_2)) * new_var * 0.5

        new_log_var = tf.math.log(new_var)

        mean_sample_1, log_var_sample_1 = self.aggregate(
            z_mean,
            z_logvar,
            new_mean,
            new_log_var,
            labels,
            kl_per_point,
        )
        mean_sample_2, log_var_sample_2 = self.aggregate(
            z_mean_2,
            z_logvar_2,
            new_mean,
            new_log_var,
            labels,
            kl_per_point,
        )

        z_sampled_1 = self.sample_from_latent_distribution(
            mean_sample_1,
            log_var_sample_1,
        )
        z_sampled_2 = self.sample_from_latent_distribution(
            mean_sample_2,
            log_var_sample_2,
        )

        reconstructions_1 = self.decoder(
            z_sampled_1,
            training=is_training
        )
        reconstructions_2 = self.decoder(
            z_sampled_2,
            training=is_training,
        )

        per_sample_loss_1 = self.loss_fn(x_1, reconstructions_1)
        per_sample_loss_2 = self.loss_fn(x_2, reconstructions_2)
        reconstruction_loss_1 = tf.reduce_mean(per_sample_loss_1)
        reconstruction_loss_2 = tf.reduce_mean(per_sample_loss_2)
        reconstruction_loss = (
            0.5 * reconstruction_loss_1 + 0.5 * reconstruction_loss_2
        )

        kl_loss_1 = compute_gaussian_kl(mean_sample_1, log_var_sample_1)
        kl_loss_2 = compute_gaussian_kl(mean_sample_2, log_var_sample_2)
        kl_loss = 0.5 * kl_loss_1 + 0.5 * kl_loss_2

        regularizer = self.regularizer(kl_loss, None, None, None)

        loss = tf.add(reconstruction_loss, regularizer, name="loss")
        elbo = tf.add(reconstruction_loss, kl_loss, name="elbo")

        return reconstruction_loss, regularizer, loss, elbo


class MLVaeLabels(MLVae):
    """Class implementing the ML-VAE with labels on which factor is shared."""

    def _split_labels(self, inputs):
        return inputs

    def aggregate(
        self,
        z_mean,
        z_logvar,
        new_mean,
        new_log_var,
        labels,
        kl_per_point,
    ):
        return aggregate_labels(
            z_mean,
            z_logvar,
            new_mean,
            new_log_var,
            labels,
            kl_per_point,
        )


class MLVaeArgmax(MLVae):
    """Class implementing the ML-VAE without any label."""

    def aggregate(
        self,
        z_mean,
        z_logvar,
        new_mean,
        new_log_var,
        labels,
        kl_per_point,
    ):
        return aggregate_argmax(
            z_mean,
            z_logvar,
            new_mean,
            new_log_var,
            kl_per_point,
        )


def aggregate_labels(
    z_mean,
    z_logvar,
    new_mean,
    new_log_var,
    labels,
    kl_per_point,
):
    """Use labels to aggregate.

    Labels contains a one-hot encoding with a single 1 of a factor shared. We
    enforce which dimension of the latent code learn which factor (dimension 1
    learns factor 1) and we enforce that each factor of variation is encoded
    in a single dimension.

    Args:
    z_mean: Mean of the encoder distribution for the original image.
    z_logvar: Logvar of the encoder distribution for the original image.
    new_mean: Average mean of the encoder distribution of the pair of images.
    new_log_var: Average logvar of the encoder distribution of the pair of
      images.
    labels: One-hot-encoding with the position of the dimension that should not
      be shared.
    kl_per_point: Distance between the two encoder distributions (unused).

    Returns:
    Mean and logvariance for the new observation.
    """
    z_mean_averaged = tf.where(
        tf.math.equal(
            labels,
            tf.expand_dims(tf.reduce_max(labels, axis=1), 1)
        ),
        z_mean,
        new_mean,
    )
    z_logvar_averaged = tf.where(
        tf.math.equal(
            labels,
            tf.expand_dims(tf.reduce_max(labels, axis=1), 1)
        ),
        z_logvar,
        new_log_var,
    )
    return z_mean_averaged, z_logvar_averaged


def aggregate_argmax(
    z_mean,
    z_logvar,
    new_mean,
    new_log_var,
    kl_per_point,
):
    """Argmax aggregation with adaptive k.

    The bottom k dimensions in terms of distance are not averaged. K is
    estimated adaptively by binning the distance into two bins of equal width.

    Args:
    z_mean: Mean of the encoder distribution for the original image.
    z_logvar: Logvar of the encoder distribution for the original image.
    new_mean: Average mean of the encoder distribution of the pair of images.
    new_log_var: Average logvar of the encoder distribution of the pair of
        images.
    kl_per_point: Distance between the two encoder distributions.

    Returns:
    Mean and logvariance for the new observation.
    """
    mask = tf.equal(tf.map_fn(discretize_in_bins, kl_per_point, tf.int32), 1)
    z_mean_averaged = tf.where(mask, z_mean, new_mean)
    z_logvar_averaged = tf.where(mask, z_logvar, new_log_var)
    return z_mean_averaged, z_logvar_averaged


def discretize_in_bins(x):
    """Discretize a vector in two bins."""
    return tf.histogram_fixed_width_bins(
        x,
        [tf.reduce_min(x), tf.reduce_max(x)],
        nbins=2,
    )


def compute_kl(z_1, z_2, logvar_1, logvar_2):
    var_1 = tf.exp(logvar_1)
    var_2 = tf.exp(logvar_2)
    return var_1/var_2 + tf.square(z_2-z_1)/var_2 - 1 + logvar_2 - logvar_1
