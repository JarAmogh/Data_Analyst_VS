from mongoengine import Document, StringField, FloatField, IntField, DateTimeField, BooleanField


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


class ExpenseReport(Document):
    received_amount = StringField()
    received_amount_value = FloatField()
    expense = StringField()
    expense_value = FloatField()

    meta = {'collection': 'expense_reports'}


class SalesReport(Document):
    sku_code = StringField()
    design_no = StringField()
    stock = IntField()
    category = StringField()
    size = StringField()
    color = StringField()

    meta = {'collection': 'sales_reports'}


class CloudWarehouseComparison(Document):
    warehouse = StringField()
    shiprocket = FloatField()
    increff = FloatField()

    meta = {'collection': 'cloud_warehouse_comparison'}


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

class May2022Report(Document):
    
    date = StringField()
    style = StringField()
    sku = StringField()
    size = StringField()
    rate = FloatField()
    gross_amt = FloatField()
    pcs = IntField()


    meta = {'collection': 'may_2022_reports'}  #

class PLMarch2021Report(Document):
    sku = StringField(required=True)
    style = StringField(required=True)
    category = StringField()
    weight = FloatField()
    tp1 = FloatField()
    tp2 = FloatField()
    mrp_old = FloatField()
    final_mrp_old = FloatField()
    ajio_mrp = FloatField()
    amazon_mrp = FloatField()
    amazon_fba_mrp = FloatField()
    flipkart_mrp = FloatField()
    limeroad_mrp = FloatField()
    myntra_mrp = FloatField()
    paytm_mrp = FloatField()
    snapdeal_mrp = FloatField()
    pcs = IntField()
    size = StringField()
    rate = StringField()
    gross_amt = FloatField()


    
