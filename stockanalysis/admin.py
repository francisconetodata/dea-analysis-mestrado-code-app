from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from .forms import CustomUsuarioChangeForm, CustomUsuarioCreateForm
from .models import CustomUsuario, LucroLiquido, PriceDataStocks, Stock


@admin.register(CustomUsuario)
class CustomUsuarioAdmin(UserAdmin):
    add_form = CustomUsuarioCreateForm
    form = CustomUsuarioChangeForm
    model = CustomUsuario
    list_display = ('first_name', 
                    'last_name', 
                    'email', 
                    'is_staff')
    fieldsets = (
        (None, {'fields': ('email', 
                           'password')}),
        ('Informações Pessoais', {
         'fields': ('first_name', 
                    'last_name')}),
        ('Permissões', 
         {'fields': ('is_active',
                     'is_staff',
         'is_superuser', 
         'groups',
         'user_permissions')}),
        ('Datas Importantes', 
         {'fields': ('last_login', 
                     'date_joined')}),
    )


admin.site.register(Stock)

admin.site.register(LucroLiquido)

admin.site.register(PriceDataStocks)


