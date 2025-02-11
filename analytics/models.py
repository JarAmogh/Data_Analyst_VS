from django.db import models
from mongoengine import Document, StringField, FloatField, DateTimeField, IntField, BooleanField

class AmazonSaleReport(Document):
    order_id = StringField(required=True, unique=True)
    date = DateTimeField()
    status = StringField()
    fulfilment = StringField()
    sales_channel = StringField()
    ship_service_level = StringField()
    style = StringField()
    sku = StringField()
    category = StringField()
    size = StringField()
    asin = StringField()
    courier_status = StringField()
    qty = FloatField()
    currency = StringField()
    amount = FloatField()
    ship_city = StringField()
    ship_state = StringField()
    ship_postal_code = StringField()
    ship_country = StringField()
    promotion_ids = StringField()
    b2b = BooleanField()
    fulfilled_by = StringField()

    meta = {'collection': 'amazon_sales'}

class SaleReport(Document):
    sku_code = StringField(required=True, unique=True)
    design_no = StringField()
    stock = IntField()
    category = StringField()
    size = StringField()
    color = StringField()

    meta = {'collection': 'sale_reports'}

class ExpenseReport(Document):
    received_amount = StringField()
    received_amount_value = FloatField()
    expense = StringField()
    expense_value = FloatField()

    meta = {'collection': 'expense_reports'}

class InternationalSaleReport(Document):
    date = DateTimeField()
    months = StringField()
    customer = StringField()
    style = StringField()
    sku = StringField()
    size = StringField()
    pcs = IntField()
    rate = FloatField()
    gross_amt = FloatField()

    meta = {'collection': 'international_sales'}

class CloudWarehouseComparison(Document):
    warehouse = StringField()
    shiprocket = FloatField()
    increff = FloatField()

    meta = {'collection': 'cloud_warehouses'}

class MonthlySalesData(Document):
    date = DateTimeField()
    order_id = StringField(unique=True)
    product = StringField()
    category = StringField()
    quantity = IntField()
    revenue = FloatField()

    meta = {'collection': 'monthly_sales'}

class ProfitLossReport(Document):
    date = DateTimeField()
    revenue = FloatField()
    expenses = FloatField()
    profit = FloatField()

    meta = {'collection': 'profit_loss'}