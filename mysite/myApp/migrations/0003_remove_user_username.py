# Generated by Django 5.1 on 2024-09-05 18:29

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('myApp', '0002_inlab_prelab_user_delete_inlabdata_prelab_user_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='user',
            name='username',
        ),
    ]
