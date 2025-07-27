"""
views.py includes the main business logic of the application.
Its role is to manage file upload, deletion and emotion predictions.
"""

import os
from os import listdir
from os.path import join
from os.path import isfile
import requests

import keras
import librosa
import numpy as np
import tensorflow as tf
from django.conf import settings
from django.views.generic import ListView
from django.views.generic import TemplateView
from django.views.generic.edit import CreateView
from rest_framework import views
from rest_framework import status
from rest_framework.generics import get_object_or_404
from rest_framework.parsers import FormParser
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework.renderers import TemplateHTMLRenderer

from App.models import FileModel
from App.serialize import FileSerializer


class IndexView(TemplateView):
    """
    This is the index view of the website.
    :param template_name; Specifies the static display template file.
    """
    template_name = 'index.html'


class FilesList(ListView):
    """
    ListView that display companies query list.
    :param model: Specifies the objects of which model we are listing
    :param template_name; Specifies the static display template file.
    :param context_object_name: Custom defined context object value,
                     this can override default context object value.
    """
    model = FileModel
    template_name = 'files_list.html'
    context_object_name = 'files_list'


class UploadView(CreateView):
    """
    This is the view that is used by the user of the web UI to upload a file.
    :param model: Specifies the objects of which model we are listing
    :param template_name; Specifies the static display template file.
    :param fields: Specifies the model field to be used
    :param success_url: Specifies the redirect url in case of successful upload.
    """
    model = FileModel
    fields = ['file']
    template_name = 'post_file.html'
    success_url = '/upload_success/'


class UploadSuccessView(TemplateView):
    """
    This is the success view of the UploadView class.
    :param template_name; Specifies the static display template file.
    """
    template_name = 'upload_success.html'


class SelectPredFileView(TemplateView):
    """
    This view is used to select a file from the list of files in the server.
    After the selection, it will send the file to the server.
    The server will return the predictions.
    """

    template_name = 'select_file_predictions.html'
    parser_classes = FormParser
    queryset = FileModel.objects.all()

    def get_context_data(self, **kwargs):
        """
        This function is used to render the list of files in the MEDIA_ROOT in the html template.
        """
        context = super().get_context_data(**kwargs)
        media_path = settings.MEDIA_ROOT
        myfiles = [f for f in listdir(media_path) if isfile(join(media_path, f))]
        context['filename'] = myfiles
        return context


class SelectFileDelView(TemplateView):
    """
    This view is used to select a file from the list of files in the server.
    After the selection, it will send the file to the server.
    The server will then delete the file.
    """
    template_name = 'select_file_deletion.html'
    parser_classes = FormParser
    queryset = FileModel.objects.all()

    def get_context_data(self, **kwargs):
        """
        This function is used to render the list of files in the MEDIA_ROOT in the html template
        and to get the pk (primary key) of each file.
        """
        context = super().get_context_data(**kwargs)
        media_path = settings.MEDIA_ROOT
        myfiles = [f for f in listdir(media_path) if isfile(join(media_path, f))]
        primary_key_list = []
        for value in myfiles:
            primary_key = FileModel.objects.filter(file=value).values_list('pk', flat=True)
            primary_key_list.append(primary_key)
        file_and_pk = zip(myfiles, primary_key_list)
        context['filename'] = file_and_pk
        return context


class FileDeleteView(views.APIView):
    """
    This class contains the method to delete a file interacting directly with the API.
    DELETE requests are accepted.
    Removing the renderer_classes an APIView instead of a TemplateView
    """
    model = FileModel
    fields = ['file']
    template_name = 'delete_success.html'
    success_url = '/delete_success/'
    renderer_classes = [TemplateHTMLRenderer]

    def post(self, request):
        """
        This method is used delete a file.
        In the identifier variable we are storing a QuerySet object.
        In the primary key object the id is extracted from the QuerySet string.
        """
        identifier = request.POST.getlist('pk').pop()
        primary_key = identifier[identifier.find("[") + 1:identifier.find("]")]
        delete_action = get_object_or_404(FileModel, pk=primary_key).delete()
        try:
            return Response({'pk': delete_action}, status=status.HTTP_200_OK)
        except ValueError as err:
            return Response(str(err), status=status.HTTP_400_BAD_REQUEST)
            


class FileView(views.APIView):
    """
    This class contains the method to upload a file interacting directly with the API.
    POST requests are accepted.
    """
    parser_classes = (MultiPartParser, FormParser)
    queryset = FileModel.objects.all()

    @staticmethod
    def upload(request):
        """
        This method is used to Make POST requests to save a file in the media folder
        """
        file_serializer = FileSerializer(data=request.data)
        if file_serializer.is_valid():
            file_serializer.save()
            response = Response(file_serializer.data, status=status.HTTP_201_CREATED)
        else:
            response = Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        return response

    @staticmethod
    def check_resource_exists(file_name):
        """
        This method will receive as input the file the user wants to store
        on the server and check if a resource (an url including
        the filename as endpoint) is existing.
        If this function returns False, the user should not be able to save the
        file (or at least he/she should be prompted with a message saying that
        the file is already existing)
        """
        request = requests.get('/media/' + file_name)
        check = bool(request.status_code == 200)
        return check

    @staticmethod
    def check_file_exists(file_name):
        """
        This method will receive as input the file the user wants to store
        on the server and check if a file with this name is physically in
        the server folder.
        If this function returns False, the user should not be able to save the
        file (or at least he/she should be prompted with a message saying that
        the file is already existing)
        """
        check = bool(str(os.path.join(settings.MEDIA_ROOT, file_name)))
        return check

    @staticmethod
    def check_object_exists(file_name):
        """
        This method will receive as input the file the user wants to store
        on the server and check if an object with that name exists in the
        database.
        If this function returns False, the user should not be able to save the
        file (or at least he/she should be prompted with a message saying that
        the file is already existing)
        """
        check = FileModel.objects.get(name=file_name).exists()
        return check


class Predict(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_path = os.path.join(settings.MODEL_ROOT, 'Emotion_Voice_Detection_Model.h5')
        self.model = load_model(model_path)

    def post(self, request, format=None):
        audio_file = request.FILES.get('file', None)
        if not audio_file:
            return Response({'error': 'No audio file provided'}, status=status.HTTP_400_BAD_REQUEST)

        # Save the uploaded file temporarily
        tmp_path = os.path.join(settings.MEDIA_ROOT, audio_file.name)
        with open(tmp_path, 'wb+') as temp_file:
            for chunk in audio_file.chunks():
                temp_file.write(chunk)

        try:
            data, sampling_rate = librosa.load(tmp_path, sr=None)
            mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
            features = np.expand_dims(mfccs, axis=2)
            features = np.expand_dims(features, axis=0)
            preds = self.model.predict(features)
            predicted_class = np.argmax(preds, axis=1)[0]
            emotion_label = self.classtoemotion(predicted_class)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        finally:
            # Clean up the temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return Response({'prediction': emotion_label}, status=status.HTTP_200_OK)

    @staticmethod
    def classtoemotion(pred):
        label_map = {
            0: 'neutral',
            1: 'calm',
            2: 'happy',
            3: 'sad',
            4: 'angry',
            5: 'fearful',
            6: 'disgust',
            7: 'surprised'
        }
        return label_map.get(pred, 'unknown')
