from __future__ import print_function
import pickle
import os.path
import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def main():
    """Based on the quickStart.py example at
    https://developers.google.com/drive/api/v3/quickstart/python
    """

    
    creds = None

     # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('drive', 'v3', credentials=creds)
    
    folderId = "1-9BAAv_P1Lr3HaMdnfs24gfMxY7sxq8T"
    destinationFolder = "models"
    downloadFolder(service, folderId, destinationFolder)


def downloadFolder(service, fileId, destinationFolder):
    if not os.path.isdir(destinationFolder):
        os.mkdir(path=destinationFolder)

    results = service.files().list(
        pageSize=300,
        q="parents in '{0}'".format(fileId),
        fields="files(id, name, mimeType)"
        ).execute()

    items = results.get('files', [])

    for item in items:
        itemName = item['name']
        itemId = item['id']
        itemType = item['mimeType']
        filePath = destinationFolder + "/" + itemName

        if itemType == 'application/vnd.google-apps.folder':
            print("Stepping into folder: {0}".format(filePath))
            downloadFolder(service, itemId, filePath) # Recursive call
        elif not itemType.startswith('application/'):
            downloadFile(service, itemId, filePath)
        elif itemType == 'application/octet-stream':
            downloadFile(service, itemId, filePath)
        else:
            print("Unsupported file: {0}".format(itemName))


def downloadFile(service, fileId, filePath):
    # Note: The parent folders in filePath must exist
    print("-> Downloading file with id: {0} name: {1}".format(fileId, filePath))
    request = service.files().get_media(fileId=fileId)
    fh = io.FileIO(filePath, mode='wb')
    
    try:
        downloader = MediaIoBaseDownload(fh, request, chunksize=1024*1024)

        done = False
        while done is False:
            status, done = downloader.next_chunk(num_retries = 2)
            if status:
                print("Download %d%%." % int(status.progress() * 100))
        print("Download Complete!")
    finally:
        fh.close()

main()