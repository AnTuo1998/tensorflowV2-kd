"""
adapt from
https://keras.io/examples/vision/knowledge_distillation/
"""
import tensorflow as tf
from tensorflow import keras

class Distiller(keras.Model):
    def __init__(self, teacher, student):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student
        self.teacher.trainable = False

    def compile(self, ce_loss, kd_loss,
                optimizer, metrics,
                alpha=0.1, temperature=4):
        super(Distiller, self).compile(
            optimizer=optimizer, metrics=metrics)
        self.ce_loss = ce_loss
        self.kd_loss = kd_loss
        self.alpha = 0.1
        self.temperature = 4

    def train_step(self, data):
        x, y = data
        y_teacher = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            y_pred = self.student(x, training=True)  # Forward pass
            ce_loss = self.ce_loss(y, y_pred)
            kd_loss = self.kd_loss(
                tf.nn.softmax(y_teacher / self.temperature, axis = 1), 
                tf.nn.softmax(y_pred / self.temperature, axis = 1)
            )
            total_loss = self.alpha * ce_loss + (1 - self.alpha) * kd_loss
            
            train_var = self.student.trainable_variables
            gradients = tape.gradient(total_loss, train_var)
            self.optimizer.apply_gradients(zip(gradients, train_var))
            self.compiled_metrics.update_state(y, y_pred)
            results = {m.name: m.result() for m in self.metrics}
            results.update(
                {"ce_loss": ce_loss, "kd_loss": kd_loss}
            )
            return results
        
        
    def test_step(self, data):
        x, y = data
        y_pred = self.student(x, training=False)
        ce_loss = self.ce_loss(y, y_pred)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"ce_loss": ce_loss})
        return results
        
