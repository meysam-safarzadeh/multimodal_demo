from django.db import models
from django.contrib.postgres.fields import ArrayField
from django.core.validators import MinValueValidator, MaxValueValidator

class Dataset(models.Model):
    name = models.CharField(max_length=200)
    csv_s3_uri = models.URLField()            # s3://... or https://...
    data_s3_uri = models.URLField()           # folder/bucket prefix
    created_at = models.DateTimeField(auto_now_add=True)

    modalities = ArrayField(models.CharField(max_length=32), default=list, blank=True)
    feature_columns = ArrayField(models.CharField(max_length=128), default=list, blank=True)
    modality_columns = ArrayField(models.CharField(max_length=128), default=list, blank=True)
    target_columns_categorical = ArrayField(models.CharField(max_length=128), default=list, blank=True)
    column_types = models.JSONField(default=dict, blank=True)
    missing_data = models.JSONField(default=dict, blank=True)
    n_rows = models.IntegerField(default=0)
    n_columns = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

class TrainingJob(models.Model):
    class Status(models.TextChoices):
        PENDING = "PENDING"
        RUNNING = "RUNNING"
        COMPLETED = "COMPLETED"
        FAILED = "FAILED"

    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name="jobs")
    job_name = models.CharField(max_length=200)
    feature_columns = ArrayField(models.CharField(max_length=128), default=list, blank=True)
    modality_columns = ArrayField(models.CharField(max_length=128), default=list, blank=True)
    target_col = models.CharField(max_length=128, null=True, blank=True)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.PENDING)
    started_at = models.DateTimeField(null=True, blank=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    ecs_task_arn = models.CharField(max_length=512, null=True, blank=True)
    epochs = models.IntegerField(default=5, validators=[MinValueValidator(1)])
    learning_rate = models.FloatField(default=0.001, validators=[MinValueValidator(1e-6), MaxValueValidator(1.0)])
    created_at = models.DateTimeField(auto_now_add=True)
    batch_size = models.IntegerField(default=16, validators=[MinValueValidator(1)])
    val_split = models.FloatField(default=0.2, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    

class TrainingMetrics(models.Model):
    job = models.OneToOneField(
        TrainingJob, on_delete=models.CASCADE, related_name="metrics"
    )
    # primary metrics
    accuracy = models.FloatField(null=True, blank=True)
    loss = models.FloatField(null=True, blank=True)
    val_accuracy = models.FloatField(null=True, blank=True)
    val_loss = models.FloatField(null=True, blank=True)
    logs = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Training Metrics"
        verbose_name_plural = "Training Metrics"

    def __str__(self):
        return f"Metrics(job={self.job.pk})"

class Artifact(models.Model):
    job = models.ForeignKey(TrainingJob, on_delete=models.CASCADE, related_name="artifacts")
    kind = models.CharField(max_length=50)                 # "model", "log", "plot"
    s3_uri = models.URLField()
    sha256 = models.CharField(max_length=100, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
