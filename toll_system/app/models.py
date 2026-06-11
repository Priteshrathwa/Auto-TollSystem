from django.db import models

class Vehicle(models.Model):
    plate = models.CharField(max_length=16, primary_key=True)
    owner_name = models.CharField(max_length=255, null=True, blank=True)
    vehicle_type = models.CharField(max_length=32, null=True, blank=True)
    fastag_id = models.CharField(max_length=64, null=True, blank=True)
    exemption = models.BooleanField(default=False)
    blacklist = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    fastag_balance = models.IntegerField(null=True, blank=True)

    class Meta:
        db_table = 'vehicles'

    def __str__(self):
        return f"{self.plate} - {self.owner_name}"

class TollRule(models.Model):
    id = models.BigAutoField(primary_key=True)
    vehicle_type = models.CharField(max_length=32)
    base_amount = models.DecimalField(max_digits=10, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'toll_rules'

    def __str__(self):
        return f"{self.vehicle_type} - ₹{self.base_amount}"

class Transaction(models.Model):
    id = models.BigAutoField(primary_key=True)
    plate = models.CharField(max_length=16, null=True, blank=True)
    toll_amount = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    payment_method = models.CharField(max_length=32, null=True, blank=True)
    payment_status = models.CharField(max_length=32, null=True, blank=True)
    location = models.CharField(max_length=128, null=True, blank=True)
    cam_id = models.CharField(max_length=64, null=True, blank=True)
    captured_at = models.DateTimeField(auto_now_add=True)
    raw_ocr_text = models.TextField(null=True, blank=True)
    ocr_confidence = models.DecimalField(max_digits=4, decimal_places=3, null=True, blank=True)
    notes = models.TextField(null=True, blank=True)

    class Meta:
        db_table = 'transactions'

    def __str__(self):
        return f"{self.plate} - ₹{self.toll_amount} - {self.captured_at}"
