# Generated by Django 4.0.3 on 2022-07-10 01:46

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('stockanalysis', '0005_alter_stock_name_stock'),
    ]

    operations = [
        migrations.CreateModel(
            name='PriceDataStock',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date_ref', models.DateField(verbose_name='Data Referência')),
                ('price_close', models.FloatField(verbose_name='Preço Fechamento Ajustado')),
                ('name_stoke', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='stockanalysis.stock')),
            ],
        ),
    ]