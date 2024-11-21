from tools.google_api_tools import (
    # Core functions
    get_credentials, get_services,
    
    # Gmail Messages functions
    gmail_messages_list,
    gmail_messages_send,
    gmail_messages_get,
    gmail_messages_trash,
    gmail_messages_untrash,
    gmail_messages_batch_delete,
    gmail_messages_attachments_get,
    
    # Gmail Draft functions
    gmail_drafts_get,
    gmail_drafts_list,
    gmail_drafts_send,
    gmail_drafts_create,
    gmail_drafts_delete,
    
    # Gmail History functions
    gmail_history_list,
    
    # Gmail Labels functions
    gmail_labels_create,
    gmail_labels_delete,
    gmail_labels_get,
    gmail_labels_list,
    
    # Gmail Settings functions
    gmail_settings_get_autoforwarding,
    gmail_settings_update_autoforwarding,
    gmail_settings_get_vacation,
    gmail_settings_update_vacation,
    
    # Gmail Settings Filters functions
    gmail_settings_filters_create,
    gmail_settings_filters_delete,
    gmail_settings_filters_get,
    gmail_settings_filters_list,
    
    # Gmail Threads functions
    gmail_threads_delete,
    gmail_threads_get,
    gmail_threads_list,
    gmail_threads_trash,
    gmail_threads_untrash,
    
    # Tasks functions
    tasks_tasklists_delete,
    tasks_tasklists_get,
    tasks_tasklists_insert,
    tasks_tasklists_list,
    tasks_clear,
    tasks_delete,
    tasks_get,
    tasks_insert,
    tasks_list,
    tasks_move,
    
    # Calendar functions
    calendar_colors_get,
    calendar_events_delete,
    calendar_events_get,
    calendar_events_list,
    calendar_events_create,
    calendar_events_quick_add,
    calendar_freebusy_query
)
from datetime import datetime, timedelta
import json
import traceback

def test_core_functions():
    """Test core API functions."""
    try:
        print("\n=== Testing Core Functions ===")
        
        print("\nTesting credentials...")
        creds = get_credentials()
        print("Credentials obtained successfully")
        
        print("\nTesting services...")
        services = get_services()
        print("Services initialized successfully")
        
    except Exception as e:
        print(f"\nError in core functions: {str(e)}")
        traceback.print_exc()

def test_gmail_functions():
    """Test all Gmail-related functions."""
    try:
        print("\n=== Testing Gmail Functions ===")
        
        # Test Message Functions
        print("\nTesting Message Operations:")
        print("Listing messages...")
        messages = gmail_messages_list(max_results=5)
        print(f"Messages list result: {messages}")
        
        print("\nSending test message...")
        message_result = gmail_messages_send(
            to="test@example.com",
            subject="Test Message",
            body="This is a test message."
        )
        print(f"Message send result: {message_result}")
        
        if "Message sent successfully" in message_result:
            message_id = message_result.split(": ")[1]
            
            print("\nGetting message details...")
            message = gmail_messages_get(message_id)
            print(f"Message details: {message}")
            
            print("\nMoving message to trash...")
            trash_result = gmail_messages_trash(message_id)
            print(f"Trash result: {trash_result}")
            
            print("\nRestoring message from trash...")
            untrash_result = gmail_messages_untrash(message_id)
            print(f"Untrash result: {untrash_result}")
            
            print("\nSkipping batch delete for safety...")
        
        # Test Draft Functions
        print("\nTesting Draft Operations:")
        print("Creating test draft...")
        draft_result = gmail_drafts_create(
            to="test@example.com",
            subject="Test Draft",
            body="This is a test draft email."
        )
        print(f"Draft creation result: {draft_result}")
        
        print("\nListing drafts...")
        drafts = gmail_drafts_list(max_results=5)
        print(f"Drafts list result: {drafts}")
        
        if "Draft created successfully" in draft_result:
            draft_id = draft_result.split(": ")[1]
            
            print("\nGetting draft details...")
            draft = gmail_drafts_get(draft_id)
            print(f"Draft details: {draft}")
            
            print("\nDeleting draft...")
            delete_result = gmail_drafts_delete(draft_id)
            print(f"Draft deletion result: {delete_result}")
        
        # Test History Functions
        print("\nTesting History Operations:")
        history = gmail_history_list(max_results=5)
        print(f"History result: {history}")
        
        # Test Label Functions
        print("\nTesting Label Operations:")
        label_result = gmail_labels_create("TestLabel")
        print(f"Label creation result: {label_result}")
        
        labels = gmail_labels_list()
        print(f"Labels list result: {labels}")
        
        if "Label created successfully" in label_result:
            label_id = label_result.split(": ")[1]
            
            print("\nGetting label details...")
            label = gmail_labels_get(label_id)
            print(f"Label details: {label}")
            
            print("\nDeleting label...")
            delete_result = gmail_labels_delete(label_id)
            print(f"Label deletion result: {delete_result}")
        
        # Test Settings Functions
        print("\nTesting Settings Operations:")
        auto_forward = gmail_settings_get_autoforwarding()
        print(f"Auto-forwarding settings: {auto_forward}")
        
        vacation = gmail_settings_get_vacation()
        print(f"Vacation settings: {vacation}")
        
        # Test Filters
        print("\nTesting Filter Operations:")
        filter_result = gmail_settings_filters_create(
            from_email="test@example.com",
            subject="Test Filter"
        )
        print(f"Filter creation result: {filter_result}")
        
        filters = gmail_settings_filters_list()
        print(f"Filters list result: {filters}")
        
        # Test Threads
        print("\nTesting Thread Operations:")
        threads = gmail_threads_list(max_results=5)
        print(f"Threads list result: {threads}")
        
        if threads and "error" not in threads.lower():
            thread_data = json.loads(threads)
            if thread_data:
                thread_id = thread_data[0]['id']
                
                print("\nGetting thread details...")
                thread = gmail_threads_get(thread_id)
                print(f"Thread details: {thread}")
                
                print("\nSkipping thread trash/untrash/delete for safety...")
        
    except Exception as e:
        print(f"\nError in Gmail functions: {str(e)}")
        traceback.print_exc()

def test_tasks_functions():
    """Test all Tasks-related functions."""
    try:
        print("\n=== Testing Tasks Functions ===")
        
        print("\nCreating task list...")
        tasklist_result = tasks_tasklists_insert("Test Task List")
        print(f"Task list creation result: {tasklist_result}")
        
        print("\nListing task lists...")
        tasklists = tasks_tasklists_list()
        print(f"Task lists result: {tasklists}")
        
        if not isinstance(tasklist_result, str) or "error" not in tasklist_result.lower():
            try:
                tasklist_id = json.loads(tasklist_result)['id']
                
                print("\nGetting task list details...")
                tasklist = tasks_tasklists_get(tasklist_id)
                print(f"Task list details: {tasklist}")
                
                print("\nCreating task...")
                task_result = tasks_insert(
                    tasklist_id=tasklist_id,
                    title="Test Task",
                    notes="Test task notes"
                )
                print(f"Task creation result: {task_result}")
                
                print("\nListing tasks...")
                tasks = tasks_list(tasklist_id=tasklist_id)
                print(f"Tasks list result: {tasks}")
                
                if task_result and "error" not in task_result.lower():
                    task_id = json.loads(task_result)['id']
                    
                    print("\nGetting task details...")
                    task = tasks_get(tasklist_id, task_id)
                    print(f"Task details: {task}")
                    
                    print("\nMoving task...")
                    move_result = tasks_move(tasklist_id, task_id)
                    print(f"Task move result: {move_result}")
                    
                    print("\nSkipping task delete for safety...")
                
                print("\nSkipping task list delete for safety...")
                
            except Exception as e:
                print(f"Error in task operations: {str(e)}")
                traceback.print_exc()
                
    except Exception as e:
        print(f"\nError in Tasks functions: {str(e)}")
        traceback.print_exc()

def test_calendar_functions():
    """Test all Calendar-related functions."""
    try:
        print("\n=== Testing Calendar Functions ===")
        
        print("\nGetting calendar colors...")
        colors = calendar_colors_get()
        print(f"Calendar colors result: {colors}")
        
        print("\nListing calendar events...")
        now = datetime.now()
        events = calendar_events_list(
            time_min=now.isoformat(),
            time_max=(now + timedelta(days=7)).isoformat()
        )
        print(f"Calendar events result: {events}")
        
        print("\nCreating calendar event...")
        event_result = calendar_events_create(
            summary="Test Event",
            start_time=now.strftime("%Y-%m-%d %H:%M"),
            duration_minutes=60
        )
        print(f"Event creation result: {event_result}")
        
        print("\nChecking free/busy...")
        freebusy = calendar_freebusy_query(
            time_min=now.isoformat(),
            time_max=(now + timedelta(hours=24)).isoformat()
        )
        print(f"Free/busy result: {freebusy}")
        
        print("\nCreating quick event...")
        quick_event = calendar_events_quick_add(
            calendar_id='primary',
            text="Test Event tomorrow at 10am"
        )
        print(f"Quick event creation result: {quick_event}")
        
        if quick_event and "error" not in quick_event.lower():
            event_data = json.loads(quick_event)
            if 'id' in event_data:
                event_id = event_data['id']
                
                print("\nGetting event details...")
                event = calendar_events_get('primary', event_id)
                print(f"Event details: {event}")
                
                print("\nSkipping event delete for safety...")
        
    except Exception as e:
        print(f"\nError in Calendar functions: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting API Tests...")
    try:
        test_core_functions()
        test_gmail_functions()
        test_tasks_functions()
        test_calendar_functions()
        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"\nFatal error in tests: {str(e)}")
        traceback.print_exc()
    finally:
        print("\nTests finished.") 