from django.shortcuts import render
import uuid
import os, fnmatch
import smtplib
import requests
import string 
import random
from rest_framework.decorators import api_view
from django.http import JsonResponse
import logging
from nlpChatbot import cursor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_ROOT = os.path.join(BASE_DIR, "static/")




# Create your views here.
@api_view(['POST'])
def forgetPwd(request):
	returnResponse = ''
	if request.method == "POST":
		text_info = request.data
		try:
			
			if text_info['AgentName'] != '' and text_info['MobileNo'] != '' and  text_info['Password'] != '':
				AgentName = text_info['AgentName']
				MobileNo = text_info['MobileNo']
				Password = text_info['Password']
				data1 = cursor.execute("EXEC [dbo].[AI_Sp_ChatAgentUser]  @Action= 'UPP', @AgentName= '"+AgentName+"', @UserName = '', @MobileNo = '"+MobileNo+"', @EmailId= '', @Password = '"+Password+"'")
				cursor.commit()
				returnResponse = {"statuscode": "100-0020", "statusmessage": "Successfully update password details"}

			else:
				returnResponse = {"statuscode": "100-0007", "statusmessage": "All field should required"}
		except Exception as e:
			logging.info(e)
	response = JsonResponse(returnResponse)
	response["Access-Control-Allow-Origin"] = "*"
	return response