# from django.contrib.auth.models import AbstractBaseUser
from operator import mod

from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.db import models


class UsuarioManager(BaseUserManager):

    use_in_migrations = True

    def _create_user(self, email, password, **extra_fields):
        if not email:
            raise ValueError('O e-mail é obrigatório')
        email = self.normalize_email(email)
        user = self.model(email=email, username=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_user(self, email, password=None, **extra_fields):
        # extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', False)
        return self._create_user(email, password, **extra_fields)

    def create_superuser(self, email, password, **extra_fields):
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('is_staff', True)

        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser precisa ter is_superuser=True')

        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser precisa ter is_staff=True')

        return self._create_user(email, password, **extra_fields)


class CustomUsuario(AbstractUser):
    email = models.EmailField('E-mail', unique=True)
    is_staff = models.BooleanField('Membro da equipe', default=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['first_name', 'last_name']

    def __str__(self):
        return self.email

    objects = UsuarioManager()


class Stock(models.Model):
    symbol_stock = models.CharField('Símbolo Empresa', max_length=10)
    name_stock = models.CharField('Nome Empresa', max_length=150)
    date_create = models.DateTimeField('Criado', auto_now_add=True)
    date_modify = models.DateTimeField('Modificado', auto_now=True)
    data_entrada = models.CharField('Data Entrada', max_length=15)
    #setor = models.CharField('Setor', max_length=55)
    def __str__(self):
        return self.symbol_stock 


    

class LucroLiquido(models.Model):
    symbol_stock = models.CharField('Símbolo Empresa', max_length=10)
    ano = models.IntegerField('Ano')
    lucro_liquido = models.FloatField('Lucro Líquido')


class PriceDataStocks(models.Model):
    date_ref = models.DateField('Data Referência')
    price_close = models.FloatField('Preço Fechamento Ajustado')
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
    
class DividendDataStocks(models.Model):
    date_ref = models.DateField('Data Referência')
    dividends = models.FloatField('Dividendos pago')
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
