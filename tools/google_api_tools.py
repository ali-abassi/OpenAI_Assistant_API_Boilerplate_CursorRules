import os
import json
import base64
from email.mime.text import MIMEText
from typing import Optional, List, Dict, Any
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from functools import lru_cache
import pickle
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv

load_dotenv()

SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/gmail.labels',
    'https://www.googleapis.com/auth/gmail.compose',
    'https://mail.google.com/',
    'https://www.googleapis.com/auth/gmail.settings.basic',
    'https://www.googleapis.com/auth/gmail.settings.sharing',
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/calendar.events',
    'https://www.googleapis.com/auth/calendar.readonly',
    'https://www.googleapis.com/auth/calendar.settings.readonly',
    'https://www.googleapis.com/auth/tasks',
    'https://www.googleapis.com/auth/tasks.readonly'
]

def get_credentials():
    """Get Google API credentials."""
    creds = None
    token_path = 'token.pickle'
    client_secret_file = os.getenv('GOOGLE_CLIENT_SECRET_FILE')
    
    if not client_secret_file or not os.path.exists(client_secret_file):
        raise ValueError(f"Client secret file not found. Please check GOOGLE_CLIENT_SECRET_FILE in .env")
    
    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)
            
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)
            
    return creds

@lru_cache(maxsize=1)
def get_services():
    """Get Gmail and Calendar services."""
    creds = get_credentials()
    return {
        'gmail': build('gmail', 'v1', credentials=creds),
        'calendar': build('calendar', 'v3', credentials=creds)
    }

def gmail_messages_list(max_results: int = 10) -> str:
    """List Gmail messages."""
    try:
        service = get_services()['gmail']
        results = service.users().messages().list(
            userId='me', maxResults=max_results, labelIds=['INBOX']
        ).execute()
        
        messages = []
        for msg in results.get('messages', []):
            message = service.users().messages().get(
                userId='me', id=msg['id'], format='metadata'
            ).execute()
            headers = message['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
            messages.append({
                'id': msg['id'],
                'subject': subject,
                'from': sender
            })
            
        return json.dumps(messages, indent=2)
    except Exception as e:
        return f"Error listing messages: {str(e)}"

def calendar_events_create(summary: str, start_time: str, duration_minutes: int = 60) -> str:
    """Create a calendar event."""
    try:
        service = get_services()['calendar']
        
        # Parse start time and calculate end time
        start = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
        end = start + timedelta(minutes=duration_minutes)
        
        timezone = 'America/New_York'
        
        event = {
            'summary': summary,
            'start': {'dateTime': start.isoformat(), 'timeZone': timezone},
            'end': {'dateTime': end.isoformat(), 'timeZone': timezone},
            'reminders': {
                'useDefault': True
            }
        }
        
        event = service.events().insert(calendarId='primary', body=event).execute()
        return f"Event created successfully: {event.get('htmlLink')}"
    except Exception as e:
        return f"Error creating event: {str(e)}"

# Gmail Draft Functions
def gmail_drafts_get(draft_id: str) -> str:
    """Get a specific draft."""
    try:
        service = get_services()['gmail']
        draft = service.users().drafts().get(userId='me', id=draft_id).execute()
        return json.dumps(draft, indent=2)
    except Exception as e:
        return f"Error getting draft: {str(e)}"

def gmail_drafts_list(max_results: int = 10) -> str:
    """List email drafts."""
    try:
        service = get_services()['gmail']
        drafts = service.users().drafts().list(userId='me', maxResults=max_results).execute()
        return json.dumps(drafts.get('drafts', []), indent=2)
    except Exception as e:
        return f"Error listing drafts: {str(e)}"

def gmail_drafts_send(draft_id: str) -> str:
    """Send an existing draft."""
    try:
        service = get_services()['gmail']
        sent = service.users().drafts().send(userId='me', body={'id': draft_id}).execute()
        return f"Draft sent successfully: {sent['id']}"
    except Exception as e:
        return f"Error sending draft: {str(e)}"

def gmail_drafts_create(to: str, subject: str, body: str) -> str:
    """Create a new draft."""
    try:
        service = get_services()['gmail']
        message = MIMEText(body)
        message['to'] = to
        message['subject'] = subject
        
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        draft = service.users().drafts().create(
            userId='me',
            body={'message': {'raw': raw}}
        ).execute()
        return f"Draft created successfully: {draft['id']}"
    except Exception as e:
        return f"Error creating draft: {str(e)}"

def gmail_drafts_delete(draft_id: str) -> str:
    """Delete a draft."""
    try:
        service = get_services()['gmail']
        service.users().drafts().delete(userId='me', id=draft_id).execute()
        return "Draft deleted successfully"
    except Exception as e:
        return f"Error deleting draft: {str(e)}"

def gmail_drafts_update(draft_id: str, to: str = None, subject: str = None, body: str = None) -> str:
    """Update an existing draft."""
    try:
        service = get_services()['gmail']
        # Get existing draft
        draft = service.users().drafts().get(userId='me', id=draft_id).execute()
        
        # Create new message
        message = MIMEText(body if body else '')
        if to:
            message['to'] = to
        if subject:
            message['subject'] = subject
        
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        updated_draft = service.users().drafts().update(
            userId='me',
            id=draft_id,
            body={'message': {'raw': raw}}
        ).execute()
        return json.dumps(updated_draft, indent=2)
    except Exception as e:
        return f"Error updating draft: {str(e)}"

# Gmail History Functions
def gmail_history_list(max_results: int = 10) -> str:
    """List history of mailbox changes."""
    try:
        service = get_services()['gmail']
        
        # First get a valid historyId from the profile
        profile = service.users().getProfile(userId='me').execute()
        start_history_id = profile.get('historyId')
        
        # Then use it to list history
        history = service.users().history().list(
            userId='me',
            startHistoryId=start_history_id,
            maxResults=max_results
        ).execute()
        
        return json.dumps(history.get('history', []), indent=2)
    except Exception as e:
        return f"Error listing history: {str(e)}"

# Gmail Labels Functions
def gmail_labels_create(name: str, label_list_visibility: str = "labelShow") -> str:
    """Create a new label."""
    try:
        service = get_services()['gmail']
        label = {
            'name': name,
            'labelListVisibility': label_list_visibility,
            'messageListVisibility': 'show'
        }
        created = service.users().labels().create(userId='me', body=label).execute()
        return f"Label created successfully: {created['id']}"
    except Exception as e:
        return f"Error creating label: {str(e)}"

def gmail_labels_delete(label_id: str) -> str:
    """Delete a label."""
    try:
        service = get_services()['gmail']
        service.users().labels().delete(userId='me', id=label_id).execute()
        return "Label deleted successfully"
    except Exception as e:
        return f"Error deleting label: {str(e)}"

def gmail_labels_get(label_id: str) -> str:
    """Get a specific label."""
    try:
        service = get_services()['gmail']
        label = service.users().labels().get(userId='me', id=label_id).execute()
        return json.dumps(label, indent=2)
    except Exception as e:
        return f"Error getting label: {str(e)}"

def gmail_labels_list() -> str:
    """List all labels."""
    try:
        service = get_services()['gmail']
        labels = service.users().labels().list(userId='me').execute()
        return json.dumps(labels.get('labels', []), indent=2)
    except Exception as e:
        return f"Error listing labels: {str(e)}"

def gmail_labels_modify(label_id: str, name: str = None, visibility: str = None) -> str:
    """Modify an existing label."""
    try:
        service = get_services()['gmail']
        body = {}
        if name:
            body['name'] = name
        if visibility:
            body['labelListVisibility'] = visibility
        
        result = service.users().labels().patch(
            userId='me',
            id=label_id,
            body=body
        ).execute()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error modifying label: {str(e)}"

# Gmail Messages Functions
def gmail_messages_get(message_id: str, format: str = 'full') -> str:
    """Get a specific message."""
    try:
        service = get_services()['gmail']
        message = service.users().messages().get(
            userId='me', id=message_id, format=format
        ).execute()
        return json.dumps(message, indent=2)
    except Exception as e:
        return f"Error getting message: {str(e)}"

def gmail_messages_send(to: str, subject: str, body: str) -> str:
    """Send a new message."""
    try:
        service = get_services()['gmail']
        message = MIMEText(body)
        message['to'] = to
        message['subject'] = subject
        
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        sent = service.users().messages().send(
            userId='me',
            body={'raw': raw}
        ).execute()
        return f"Message sent successfully: {sent['id']}"
    except Exception as e:
        return f"Error sending message: {str(e)}"

def gmail_messages_trash(message_id: str) -> str:
    """Move a message to trash."""
    try:
        service = get_services()['gmail']
        service.users().messages().trash(userId='me', id=message_id).execute()
        return "Message moved to trash successfully"
    except Exception as e:
        return f"Error moving message to trash: {str(e)}"

def gmail_messages_untrash(message_id: str) -> str:
    """Remove a message from trash."""
    try:
        service = get_services()['gmail']
        service.users().messages().untrash(userId='me', id=message_id).execute()
        return "Message removed from trash successfully"
    except Exception as e:
        return f"Error removing message from trash: {str(e)}"

def gmail_messages_batch_delete(message_ids: List[str]) -> str:
    """Delete multiple messages."""
    try:
        service = get_services()['gmail']
        service.users().messages().batchDelete(
            userId='me',
            body={'ids': message_ids}
        ).execute()
        return "Messages deleted successfully"
    except Exception as e:
        return f"Error deleting messages: {str(e)}"

def gmail_messages_attachments_get(message_id: str, attachment_id: str) -> str:
    """Get a message attachment."""
    try:
        service = get_services()['gmail']
        attachment = service.users().messages().attachments().get(
            userId='me',
            messageId=message_id,
            id=attachment_id
        ).execute()
        return json.dumps(attachment, indent=2)
    except Exception as e:
        return f"Error getting attachment: {str(e)}"

# Gmail Settings Functions
def gmail_settings_get_autoforwarding() -> str:
    """Get auto-forwarding settings."""
    try:
        service = get_services()['gmail']
        result = service.users().settings().getAutoForwarding(userId='me').execute()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error getting auto-forwarding settings: {str(e)}"

def gmail_settings_update_autoforwarding(enabled: bool, email: str = None) -> str:
    """Update auto-forwarding settings."""
    try:
        service = get_services()['gmail']
        body = {
            'enabled': enabled,
            'emailAddress': email if enabled else None
        }
        result = service.users().settings().updateAutoForwarding(
            userId='me', body=body
        ).execute()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error updating auto-forwarding settings: {str(e)}"

def gmail_settings_get_vacation() -> str:
    """Get vacation responder settings."""
    try:
        service = get_services()['gmail']
        result = service.users().settings().getVacation(userId='me').execute()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error getting vacation settings: {str(e)}"

def gmail_settings_update_vacation(
    enabled: bool,
    response_subject: str = "",
    response_body: str = "",
    start_time: str = None,
    end_time: str = None
) -> str:
    """Update vacation responder settings."""
    try:
        service = get_services()['gmail']
        body = {
            'enableAutoReply': enabled,
            'responseSubject': response_subject,
            'responseBodyHtml': response_body,
        }
        if start_time:
            body['startTime'] = start_time
        if end_time:
            body['endTime'] = end_time
            
        result = service.users().settings().updateVacation(
            userId='me', body=body
        ).execute()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error updating vacation settings: {str(e)}"

# Gmail Settings Filters Functions
def gmail_settings_filters_create(
    from_email: str = None,
    to_email: str = None,
    subject: str = None,
    has_words: str = None,
    label_ids: List[str] = None
) -> str:
    """Create a Gmail filter."""
    try:
        service = get_services()['gmail']
        filter_criteria = {}
        if from_email:
            filter_criteria['from'] = from_email
        if to_email:
            filter_criteria['to'] = to_email
        if subject:
            filter_criteria['subject'] = subject
        if has_words:
            filter_criteria['query'] = has_words
            
        filter_action = {}
        if label_ids:
            filter_action['addLabelIds'] = label_ids
            
        body = {
            'criteria': filter_criteria,
            'action': filter_action
        }
        
        result = service.users().settings().filters().create(
            userId='me', body=body
        ).execute()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error creating filter: {str(e)}"

def gmail_settings_filters_delete(filter_id: str) -> str:
    """Delete a Gmail filter."""
    try:
        service = get_services()['gmail']
        service.users().settings().filters().delete(
            userId='me', id=filter_id
        ).execute()
        return "Filter deleted successfully"
    except Exception as e:
        return f"Error deleting filter: {str(e)}"

def gmail_settings_filters_get(filter_id: str) -> str:
    """Get a specific Gmail filter."""
    try:
        service = get_services()['gmail']
        result = service.users().settings().filters().get(
            userId='me', id=filter_id
        ).execute()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error getting filter: {str(e)}"

def gmail_settings_filters_list() -> str:
    """List all Gmail filters."""
    try:
        service = get_services()['gmail']
        results = service.users().settings().filters().list(userId='me').execute()
        return json.dumps(results.get('filter', []), indent=2)
    except Exception as e:
        return f"Error listing filters: {str(e)}"

# Gmail Threads Functions
def gmail_threads_delete(thread_id: str) -> str:
    """Delete a thread."""
    try:
        service = get_services()['gmail']
        service.users().threads().delete(userId='me', id=thread_id).execute()
        return "Thread deleted successfully"
    except Exception as e:
        return f"Error deleting thread: {str(e)}"

def gmail_threads_get(thread_id: str) -> str:
    """Get a specific thread."""
    try:
        service = get_services()['gmail']
        result = service.users().threads().get(userId='me', id=thread_id).execute()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error getting thread: {str(e)}"

def gmail_threads_list(max_results: int = 10) -> str:
    """List email threads."""
    try:
        service = get_services()['gmail']
        results = service.users().threads().list(
            userId='me', maxResults=max_results
        ).execute()
        return json.dumps(results.get('threads', []), indent=2)
    except Exception as e:
        return f"Error listing threads: {str(e)}"

def gmail_threads_trash(thread_id: str) -> str:
    """Move a thread to trash."""
    try:
        service = get_services()['gmail']
        service.users().threads().trash(userId='me', id=thread_id).execute()
        return "Thread moved to trash successfully"
    except Exception as e:
        return f"Error moving thread to trash: {str(e)}"

def gmail_threads_untrash(thread_id: str) -> str:
    """Remove a thread from trash."""
    try:
        service = get_services()['gmail']
        service.users().threads().untrash(userId='me', id=thread_id).execute()
        return "Thread removed from trash successfully"
    except Exception as e:
        return f"Error removing thread from trash: {str(e)}"

# Tasks Functions
def get_tasks_service():
    """Get Tasks API service."""
    creds = get_credentials()
    return build('tasks', 'v1', credentials=creds)

def tasks_tasklists_delete(tasklist_id: str) -> str:
    """Delete a task list."""
    try:
        service = get_tasks_service()
        service.tasklists().delete(tasklist=tasklist_id).execute()
        return "Task list deleted successfully"
    except Exception as e:
        return f"Error deleting task list: {str(e)}"

def tasks_tasklists_get(tasklist_id: str) -> str:
    """Get a specific task list."""
    try:
        service = get_tasks_service()
        result = service.tasklists().get(tasklist=tasklist_id).execute()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error getting task list: {str(e)}"

def tasks_tasklists_insert(title: str) -> str:
    """Create a new task list."""
    try:
        service = get_tasks_service()
        result = service.tasklists().insert(body={'title': title}).execute()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error creating task list: {str(e)}"

def tasks_tasklists_list() -> str:
    """List all task lists."""
    try:
        service = get_tasks_service()
        results = service.tasklists().list().execute()
        return json.dumps(results.get('items', []), indent=2)
    except Exception as e:
        return f"Error listing task lists: {str(e)}"

def tasks_clear(tasklist_id: str) -> str:
    """Clear all completed tasks."""
    try:
        service = get_tasks_service()
        service.tasks().clear(tasklist=tasklist_id).execute()
        return "Completed tasks cleared successfully"
    except Exception as e:
        return f"Error clearing tasks: {str(e)}"

def tasks_delete(tasklist_id: str, task_id: str) -> str:
    """Delete a task."""
    try:
        service = get_tasks_service()
        service.tasks().delete(tasklist=tasklist_id, task=task_id).execute()
        return json.dumps({
            'status': 'success',
            'message': f"Task {task_id} deleted successfully",
            'tasklist_id': tasklist_id,
            'task_id': task_id
        })
    except Exception as e:
        return json.dumps({
            'status': 'error',
            'message': f"Error deleting task {task_id}: {str(e)}",
            'tasklist_id': tasklist_id,
            'task_id': task_id
        })

def tasks_get(tasklist_id: str, task_id: str) -> str:
    """Get a specific task."""
    try:
        service = get_tasks_service()
        result = service.tasks().get(tasklist=tasklist_id, task=task_id).execute()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error getting task: {str(e)}"

def tasks_insert(
    tasklist_id: str,
    title: str,
    notes: str = "",
    due: str = None
) -> str:
    """Create a new task."""
    try:
        service = get_tasks_service()
        body = {
            'title': title,
            'notes': notes
        }
        if due:
            body['due'] = due
            
        result = service.tasks().insert(tasklist=tasklist_id, body=body).execute()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error creating task: {str(e)}"

def tasks_list(tasklist_id: str, max_results: int = 100) -> str:
    """List tasks in a task list."""
    try:
        service = get_tasks_service()
        results = service.tasks().list(
            tasklist=tasklist_id,
            maxResults=max_results,
            showCompleted=True,
            showDeleted=False,
            showHidden=True
        ).execute()
        
        # Format tasks to ensure IDs are prominently displayed
        formatted_tasks = []
        for index, task in enumerate(results.get('items', []), 1):
            formatted_tasks.append({
                'index': index,
                'task_id': task['id'],  # Make task ID prominent
                'title': task.get('title', 'No Title'),
                'notes': task.get('notes', ''),
                'status': task.get('status', 'needsAction'),
                'due': task.get('due', 'Not specified')
            })
            
        # Return a more structured response
        response = {
            'tasklist_id': tasklist_id,
            'task_count': len(formatted_tasks),
            'tasks': formatted_tasks
        }
        
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error listing tasks: {str(e)}"

def tasks_move(
    tasklist_id: str,
    task_id: str,
    parent: str = None,
    previous: str = None
) -> str:
    """Move a task."""
    try:
        service = get_tasks_service()
        result = service.tasks().move(
            tasklist=tasklist_id,
            task=task_id,
            parent=parent,
            previous=previous
        ).execute()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error moving task: {str(e)}"

# Calendar Functions
def calendar_colors_get() -> str:
    """Get calendar color definitions."""
    try:
        service = get_services()['calendar']
        result = service.colors().get().execute()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error getting calendar colors: {str(e)}"

def calendar_events_delete(calendar_id: str, event_id: str) -> str:
    """Delete a calendar event."""
    try:
        service = get_services()['calendar']
        service.events().delete(calendarId=calendar_id, eventId=event_id).execute()
        return "Event deleted successfully"
    except Exception as e:
        return f"Error deleting event: {str(e)}"

def calendar_events_get(calendar_id: str, event_id: str) -> str:
    """Get a specific calendar event."""
    try:
        service = get_services()['calendar']
        result = service.events().get(calendarId=calendar_id, eventId=event_id).execute()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error getting event: {str(e)}"

def calendar_events_list(
    calendar_id: str = 'primary',
    max_results: int = 10,
    time_min: str = None,
    time_max: str = None
) -> str:
    """List calendar events."""
    try:
        service = get_services()['calendar']
        params = {
            'calendarId': calendar_id,
            'maxResults': max_results,
            'orderBy': 'startTime',
            'singleEvents': True
        }
        if time_min:
            params['timeMin'] = time_min
        if time_max:
            params['timeMax'] = time_max
            
        results = service.events().list(**params).execute()
        return json.dumps(results.get('items', []), indent=2)
    except Exception as e:
        return f"Error listing events: {str(e)}"

def calendar_events_quick_add(
    calendar_id: str,
    text: str
) -> str:
    """Quickly add an event from text."""
    try:
        service = get_services()['calendar']
        result = service.events().quickAdd(
            calendarId=calendar_id,
            text=text
        ).execute()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error creating quick event: {str(e)}"

def calendar_freebusy_query(
    time_min: str,
    time_max: str,
    calendar_ids: List[str] = None
) -> str:
    """Query free/busy information."""
    try:
        service = get_services()['calendar']
        body = {
            'timeMin': time_min,
            'timeMax': time_max,
            'items': [{'id': cal_id} for cal_id in (calendar_ids or ['primary'])]
        }
        result = service.freebusy().query(body=body).execute()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error querying free/busy: {str(e)}"

def gmail_messages_create(to: str, subject: str, body: str, draft: bool = False) -> str:
    """Create a new message (as draft or sent)."""
    try:
        service = get_services()['gmail']
        message = MIMEText(body)
        message['to'] = to
        message['subject'] = subject
        
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        if draft:
            result = service.users().drafts().create(
                userId='me',
                body={'message': {'raw': raw}}
            ).execute()
        else:
            result = service.users().messages().send(
                userId='me',
                body={'raw': raw}
            ).execute()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error creating message: {str(e)}"

def gmail_messages_delete(message_id: str) -> str:
    """Permanently delete a message."""
    try:
        service = get_services()['gmail']
        service.users().messages().delete(userId='me', id=message_id).execute()
        return json.dumps({
            'status': 'success',
            'message': f"Message {message_id} deleted successfully"
        })
    except Exception as e:
        return json.dumps({
            'status': 'error',
            'message': f"Error deleting message {message_id}: {str(e)}"
        })

def gmail_messages_import(raw_email: str, labels: List[str] = None) -> str:
    """Import a raw message."""
    try:
        service = get_services()['gmail']
        message = {'raw': base64.urlsafe_b64encode(raw_email.encode()).decode()}
        if labels:
            message['labelIds'] = labels
        
        result = service.users().messages().import_(
            userId='me',
            body=message
        ).execute()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error importing message: {str(e)}"

# Calendar Functions (additions)
def calendar_events_instances(calendar_id: str, event_id: str, max_results: int = 10) -> str:
    """Get instances of a recurring event."""
    try:
        service = get_services()['calendar']
        results = service.events().instances(
            calendarId=calendar_id,
            eventId=event_id,
            maxResults=max_results
        ).execute()
        return json.dumps(results.get('items', []), indent=2)
    except Exception as e:
        return f"Error getting event instances: {str(e)}"

def calendar_events_move(calendar_id: str, event_id: str, destination_id: str) -> str:
    """Move an event to a different calendar."""
    try:
        service = get_services()['calendar']
        result = service.events().move(
            calendarId=calendar_id,
            eventId=event_id,
            destination=destination_id
        ).execute()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error moving event: {str(e)}"

def calendar_events_patch(calendar_id: str, event_id: str, **kwargs) -> str:
    """Update specific fields of an event."""
    try:
        service = get_services()['calendar']
        # Build the patch body from kwargs
        body = {}
        if 'summary' in kwargs:
            body['summary'] = kwargs['summary']
        if 'description' in kwargs:
            body['description'] = kwargs['description']
        if 'start' in kwargs:
            body['start'] = kwargs['start']
        if 'end' in kwargs:
            body['end'] = kwargs['end']
        
        result = service.events().patch(
            calendarId=calendar_id,
            eventId=event_id,
            body=body
        ).execute()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error patching event: {str(e)}"

def calendar_events_update(calendar_id: str, event_id: str, event_data: Dict[str, Any]) -> str:
    """Replace all fields of an event."""
    try:
        service = get_services()['calendar']
        result = service.events().update(
            calendarId=calendar_id,
            eventId=event_id,
            body=event_data
        ).execute()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error updating event: {str(e)}"

def tasks_tasklists_patch(tasklist_id: str, title: str) -> str:
    """Update a task list's metadata."""
    try:
        service = get_tasks_service()
        result = service.tasklists().patch(
            tasklist=tasklist_id,
            body={'title': title}
        ).execute()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error patching task list: {str(e)}"

def tasks_tasklists_update(tasklist_id: str, title: str) -> str:
    """Replace a task list's metadata."""
    try:
        service = get_tasks_service()
        result = service.tasklists().update(
            tasklist=tasklist_id,
            body={'title': title}
        ).execute()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error updating task list: {str(e)}"

def tasks_patch(tasklist_id: str, task_id: str, **kwargs) -> str:
    """Update specific fields of a task."""
    try:
        service = get_tasks_service()
        body = {}
        if 'title' in kwargs:
            body['title'] = kwargs['title']
        if 'notes' in kwargs:
            body['notes'] = kwargs['notes']
        if 'due' in kwargs:
            body['due'] = kwargs['due']
        
        result = service.tasks().patch(
            tasklist=tasklist_id,
            task=task_id,
            body=body
        ).execute()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error patching task: {str(e)}"

def tasks_update(tasklist_id: str, task_id: str, title: str, notes: str = "", due: str = None) -> str:
    """Replace all fields of a task."""
    try:
        service = get_tasks_service()
        body = {
            'title': title,
            'notes': notes
        }
        if due:
            body['due'] = due
            
        result = service.tasks().update(
            tasklist=tasklist_id,
            task=task_id,
            body=body
        ).execute()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error updating task: {str(e)}" 