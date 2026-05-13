from django import forms

class DataInput(forms.Form):
    """
    Class đại diện cho đối tượng dữ liệu đầu vào.
    Sử dụng để validate dữ liệu từ request.POST trước khi đưa vào model ML.
    """
    
    # Số lần mang thai
    pregnancies = forms.IntegerField(min_value=0)
    
    # Chỉ số Glucose
    glucose = forms.FloatField(min_value=0)
    
    # Huyết áp
    blood_pressure = forms.FloatField(min_value=0)
    
    # Độ dày lớp da
    skin_thickness = forms.FloatField(min_value=0)
    
    # Chỉ số Insulin
    insulin = forms.FloatField(min_value=0)
    
    # Chỉ số khối cơ thể (BMI)
    bmi = forms.FloatField(min_value=0)
    
    # Chức năng phả hệ bệnh tiểu đường
    diabetes_pedigree_function = forms.FloatField(min_value=0)
    # Tuổi
    age = forms.IntegerField(min_value=0)

    def __str__(self):
        return f"Patient Data: Age={self.data.get('age')}, BMI={self.data.get('bmi')}"