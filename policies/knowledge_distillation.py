
import tensorflow as tf

def distil_knowledge(
        *
        obs,
        prev_action,
        prev_reward,
        state,
        seq_lens,
        student_logits,
        teacher,
        mask,
        temperature,
):
    teacher_logits, _ = teacher(
        obs=obs,
        prev_action=prev_action,
        prev_reward=prev_reward,
        state=state,
        seq_lens=seq_lens
    )

    teacher_dist = teacher.action_dist(tf.stop_gradient(teacher_logits))
    kl_loss = tf.reduce_mean(tf.boolean_mask(teacher_dist.kl(student_logits), mask)) * tf.math.square(temperature)
    return kl_loss
