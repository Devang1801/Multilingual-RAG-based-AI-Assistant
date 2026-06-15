
def login_token():
        url="http://192.168.1.52:4050/auth/chat/sign-in&quot;
        headers={"Content-Type":"application/json"
                        }
        body={
        "mobile":"7668455121",
        "otp" : "000000",
        "fp" : "test-device"
        }
        token=requests.post(url,headers=headers,json=body)
        login_token=token.json()['message']
        return login_token
        
        
def data_fetch():
        token=login_token()
        headers={"Content-Type":"application/json",
                          "Authorization": f"Bearer {token}",
                        }
        url="http://192.168.1.52:4050/chatbot/candidate_details_internship&quot;
        data=requests.post(url=url,headers=headers)
        print(data.json())
        
        
data_fetch()