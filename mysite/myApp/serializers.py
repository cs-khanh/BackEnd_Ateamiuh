from rest_framework import serializers
from .models import User,PreLab,InLab,QuestionData,FinalData,predictFinalUser

      
class PrelabSerializer(serializers.ModelSerializer):
    class Meta:
        model = PreLab
        fields = '__all__'

class InLabSerializer(serializers.ModelSerializer):
    class Meta:
        model = InLab
        fields = '__all__'
        
class UserSerializer(serializers.ModelSerializer):
    prelabs = PrelabSerializer(many=True, required=False)
    inlabs = InLabSerializer(many=True, required=False)
    class Meta:
        model = User
        fields = '__all__'
        
    def create(self, validated_data):
        prelabs_data = validated_data.pop('prelabs', [])
        inlabs_data = validated_data.pop('inlabs', [])
        user = User.objects.create(**validated_data)
        for prelab_data in prelabs_data:
            PreLab.objects.create(user=user, **prelab_data)
        for inlab_data in inlabs_data:
            InLab.objects.create(user=user, **inlab_data)
        return user
    
class QuestionDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = QuestionData
        fields='__all__'
        
class FinalDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = FinalData
        fields='__all__'

class predictFinalUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = predictFinalUser
        fields='__all__'