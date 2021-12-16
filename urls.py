"""nlpChatbot URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import include, url
from textTokenize.views import textTokenizeService
from wordSentenceMatch.views import sentenceMatchService
from otpGenerator.views import botOTPGenerator
from otpValidator.views import botOTPValidator
from chatAgentUser.views import botAgentUser
from pwdValidate.views import botpwdValidate
from forgetPassword.views import forgetPwd
from intentCreator.views import intentCreatorService
from welcomeIntent.views import welcomeIntentService
from chatbotDetails.views import chatbotdetailsService
from singlesignup.views import singleSignUp

urlpatterns = [
    path('admin/', admin.site.urls),
    url('text_tokenize',textTokenizeService),
    url('sentence_match',sentenceMatchService),
    url('bot_otp_generator',botOTPGenerator),
    url('bot_otp_validator',botOTPValidator),
    url('chat_agent_user',botAgentUser),
    url('chat_pwd_validate',botpwdValidate),
    url('forgetpassword',forgetPwd),
    url('botIntent',intentCreatorService),
    url('welcome_intent',welcomeIntentService),
    url('chatbot_details',chatbotdetailsService),
    url('singleSignUp',singleSignUp),
]
