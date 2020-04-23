from numbers import Number

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.losses import huber_loss
from .sac import SAC, td_target


class TQC(SAC):
    def __init__(self, num_quantiles, top_crop_quantiles, p_opt_crop, **kwargs):
        self.num_quantiles = num_quantiles
        self.top_crop_quantiles = top_crop_quantiles
        self.total_top_crop = top_crop_quantiles * kwargs['num_q']
        self.total_quantiles = num_quantiles * kwargs['num_q']
        self.total_active_quantiles = self.total_quantiles - self.total_top_crop
        self.p_opt_crop = p_opt_crop
        super(TQC, self).__init__(**kwargs)

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.

        See Equations (5, 6) in [1], for further information of the
        Q-function update rule.
        """
        self.tau = ((2 * tf.range(self.num_quantiles, dtype='float32') + 1) / (2 * self.num_quantiles))[None, None, :]
        Q_target = tf.stop_gradient(self._get_Q_target())

        Q_values = self._Q_values = tuple(
            Q([self._observations_ph, self._actions_ph])
            for Q in self._Qs)
        Q_target = tf.tile(Q_target[:, :, None], (1, 1, self.num_quantiles))
        Q_values = tuple(tf.tile(Q_value[:, None, :], (1, self.total_active_quantiles, 1)) for Q_value in Q_values)
        Q_losses = tuple(huber_loss(Q_target, Q_value, reduction='none') * tf.math.abs(
            self.tau - tf.cast((Q_target < Q_value), 'float32')) for Q_value in Q_values)
        Q_losses = self._Q_losses = tuple(tf.reduce_mean(loss) for loss in Q_losses)

        self._Q_optimizers = tuple(
            tf.train.AdamOptimizer(
                learning_rate=self._Q_lr,
                name='{}_{}_optimizer'.format(Q._name, i)
            ) for i, Q in enumerate(self._Qs))

        Q_training_ops = tuple(
            Q_optimizer.minimize(loss=Q_loss, var_list=Q.trainable_variables)
            for i, (Q, Q_loss, Q_optimizer)
            in enumerate(zip(self._Qs, Q_losses, self._Q_optimizers)))

        self._training_ops.update({'Q': tf.group(Q_training_ops)})

    def _get_Q_target(self):
        next_actions = self._policy.actions([self._next_observations_ph])
        next_log_pis = self._policy.log_pis([self._next_observations_ph], next_actions)

        next_Qs_values = tuple(
            Q([self._next_observations_ph, next_actions])
            for Q in self._Q_targets)

        next_Qs_values = tf.concat(next_Qs_values, axis=1)
        min_next_Q = tf.sort(next_Qs_values, axis=1)[:, :self.total_active_quantiles]
        next_value = min_next_Q - self._alpha * next_log_pis

        Q_target = td_target(
            reward=self._reward_scale * self._rewards_ph,
            discount=self._discount,
            next_value=(1 - self._terminals_ph) * next_value)

        return Q_target

    def _init_actor_update(self):
        """Create minimization operations for policy and entropy.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.

        See Section 4.2 in [1], for further information of the policy update,
        and Section 5 in [1] for further information of the entropy update.
        """

        actions = self._policy.actions([self._observations_ph])
        log_pis = self._policy.log_pis([self._observations_ph], actions)

        assert log_pis.shape.as_list() == [None, 1]

        log_alpha = self._log_alpha = tf.get_variable(
            'log_alpha',
            dtype=tf.float32,
            initializer=0.0)
        alpha = tf.exp(log_alpha)

        if isinstance(self._target_entropy, Number):
            alpha_loss = -tf.reduce_mean(
                log_alpha * tf.stop_gradient(log_pis + self._target_entropy))

            self._alpha_optimizer = tf.train.AdamOptimizer(
                self._policy_lr, name='alpha_optimizer')
            self._alpha_train_op = self._alpha_optimizer.minimize(
                loss=alpha_loss, var_list=[log_alpha])

            self._training_ops.update({
                'temperature_alpha': self._alpha_train_op
            })

        self._alpha = alpha

        if self._action_prior == 'normal':
            policy_prior = tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros(self._action_shape),
                scale_diag=tf.ones(self._action_shape))
            policy_prior_log_probs = policy_prior.log_prob(actions)
        elif self._action_prior == 'uniform':
            policy_prior_log_probs = 0.0

        Q_log_targets = tuple(Q([self._observations_ph, actions])
            for Q in self._Qs)
        Q_log_targets = tf.concat(Q_log_targets, axis=1)
        if self.p_opt_crop:
            Q_log_targets = tf.sort(Q_log_targets, axis=1)[:, :self.total_active_quantiles]
        min_Q_log_target = tf.reduce_mean(Q_log_targets, axis=1, keepdims=True)

        if self._reparameterize:
            policy_kl_losses = (
                alpha * log_pis
                - min_Q_log_target
                - policy_prior_log_probs)
        else:
            raise NotImplementedError

        assert policy_kl_losses.shape.as_list() == [None, 1]

        self._policy_losses = policy_kl_losses
        policy_loss = tf.reduce_mean(policy_kl_losses)

        self._policy_optimizer = tf.train.AdamOptimizer(
            learning_rate=self._policy_lr,
            name="policy_optimizer")

        policy_train_op = self._policy_optimizer.minimize(
            loss=policy_loss,
            var_list=self._policy.trainable_variables)

        self._training_ops.update({'policy_train_op': policy_train_op})