def get_tool_definitions():
    """Return the list of tool definitions for the assistant."""
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file from the working directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The name of the file to read"
                        }
                    },
                    "required": ["file_path"],
                    "additionalProperties": False
                },
                "strict": True
            }
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file in the working directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The name of the file to write"
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to write to the file"
                        }
                    },
                    "required": ["file_path", "content"],
                    "additionalProperties": False
                },
                "strict": True
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_messages_list",
                "description": "List recent emails from Gmail inbox",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of emails to return"
                        }
                    },
                    "required": [],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calendar_events_create",
                "description": "Create a Google Calendar event. Date must be in format 'YYYY-MM-DD HH:MM' (e.g., '2024-03-14 15:30')",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Event title"
                        },
                        "start_time": {
                            "type": "string",
                            "description": "Start time in format 'YYYY-MM-DD HH:MM' (e.g., '2024-03-14 15:30')"
                        },
                        "duration_minutes": {
                            "type": "integer",
                            "description": "Event duration in minutes (default: 60)"
                        }
                    },
                    "required": ["summary", "start_time"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_drafts_get",
                "description": "Get a specific Gmail draft",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "draft_id": {
                            "type": "string",
                            "description": "ID of the draft to retrieve"
                        }
                    },
                    "required": ["draft_id"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_drafts_list",
                "description": "List Gmail drafts",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of drafts to return (default: 10)"
                        }
                    },
                    "required": [],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_drafts_send",
                "description": "Send an existing Gmail draft",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "draft_id": {
                            "type": "string",
                            "description": "ID of the draft to send"
                        }
                    },
                    "required": ["draft_id"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_drafts_create",
                "description": "Create a new Gmail draft",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": "string",
                            "description": "Recipient email address"
                        },
                        "subject": {
                            "type": "string",
                            "description": "Email subject"
                        },
                        "body": {
                            "type": "string",
                            "description": "Email body content"
                        }
                    },
                    "required": ["to", "subject", "body"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_drafts_delete",
                "description": "Delete a Gmail draft",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "draft_id": {
                            "type": "string",
                            "description": "ID of the draft to delete"
                        }
                    },
                    "required": ["draft_id"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_history_list",
                "description": "List Gmail mailbox history changes",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_history_id": {
                            "type": "string",
                            "description": "Starting point for history (optional)"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of history records to return (default: 10)"
                        }
                    },
                    "required": [],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_labels_create",
                "description": "Create a new Gmail label",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the new label"
                        },
                        "label_list_visibility": {
                            "type": "string",
                            "description": "Label visibility setting (default: 'labelShow')",
                            "enum": ["labelShow", "labelHide", "labelShowIfUnread"]
                        }
                    },
                    "required": ["name"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_labels_delete",
                "description": "Delete a Gmail label",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "label_id": {
                            "type": "string",
                            "description": "ID of the label to delete"
                        }
                    },
                    "required": ["label_id"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_labels_get",
                "description": "Get details of a specific Gmail label",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "label_id": {
                            "type": "string",
                            "description": "ID of the label to retrieve"
                        }
                    },
                    "required": ["label_id"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_labels_list",
                "description": "List all Gmail labels",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_messages_get",
                "description": "Get a specific Gmail message",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message_id": {
                            "type": "string",
                            "description": "ID of the message to retrieve"
                        },
                        "format": {
                            "type": "string",
                            "description": "Format of the message (default: 'full')",
                            "enum": ["minimal", "full", "raw", "metadata"]
                        }
                    },
                    "required": ["message_id"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_messages_send",
                "description": "Send a new Gmail message",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": "string",
                            "description": "Recipient email address"
                        },
                        "subject": {
                            "type": "string",
                            "description": "Email subject"
                        },
                        "body": {
                            "type": "string",
                            "description": "Email body content"
                        }
                    },
                    "required": ["to", "subject", "body"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_messages_trash",
                "description": "Move a Gmail message to trash",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message_id": {
                            "type": "string",
                            "description": "ID of the message to move to trash"
                        }
                    },
                    "required": ["message_id"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_messages_untrash",
                "description": "Remove a Gmail message from trash",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message_id": {
                            "type": "string",
                            "description": "ID of the message to remove from trash"
                        }
                    },
                    "required": ["message_id"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_messages_batch_delete",
                "description": "Delete multiple Gmail messages permanently",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message_ids": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "List of message IDs to delete"
                        }
                    },
                    "required": ["message_ids"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_messages_attachments_get",
                "description": "Get an attachment from a Gmail message",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message_id": {
                            "type": "string",
                            "description": "ID of the message containing the attachment"
                        },
                        "attachment_id": {
                            "type": "string",
                            "description": "ID of the attachment to retrieve"
                        }
                    },
                    "required": ["message_id", "attachment_id"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_settings_get_autoforwarding",
                "description": "Get Gmail auto-forwarding settings",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_settings_update_autoforwarding",
                "description": "Update Gmail auto-forwarding settings",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "enabled": {
                            "type": "boolean",
                            "description": "Whether auto-forwarding is enabled"
                        },
                        "email": {
                            "type": "string",
                            "description": "Email address to forward to (required if enabled is true)"
                        }
                    },
                    "required": ["enabled"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_settings_get_vacation",
                "description": "Get Gmail vacation responder settings",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_settings_update_vacation",
                "description": "Update Gmail vacation responder settings",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "enabled": {
                            "type": "boolean",
                            "description": "Whether vacation responder is enabled"
                        },
                        "response_subject": {
                            "type": "string",
                            "description": "Subject line of auto-reply"
                        },
                        "response_body": {
                            "type": "string",
                            "description": "Body of auto-reply message"
                        },
                        "start_time": {
                            "type": "string",
                            "description": "Start time in ISO format (optional)"
                        },
                        "end_time": {
                            "type": "string",
                            "description": "End time in ISO format (optional)"
                        }
                    },
                    "required": ["enabled"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_settings_filters_create",
                "description": "Create a new Gmail filter",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from_email": {
                            "type": "string",
                            "description": "Filter emails from this address"
                        },
                        "to_email": {
                            "type": "string",
                            "description": "Filter emails to this address"
                        },
                        "subject": {
                            "type": "string",
                            "description": "Filter emails with this subject"
                        },
                        "has_words": {
                            "type": "string",
                            "description": "Filter emails containing these words"
                        },
                        "label_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Labels to apply to matching emails"
                        }
                    },
                    "required": [],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_settings_filters_delete",
                "description": "Delete a Gmail filter",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filter_id": {
                            "type": "string",
                            "description": "ID of the filter to delete"
                        }
                    },
                    "required": ["filter_id"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_settings_filters_get",
                "description": "Get a specific Gmail filter",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filter_id": {
                            "type": "string",
                            "description": "ID of the filter to retrieve"
                        }
                    },
                    "required": ["filter_id"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_settings_filters_list",
                "description": "List all Gmail filters",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_threads_list",
                "description": "List Gmail threads",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of threads to return (default: 10)"
                        }
                    },
                    "required": [],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_threads_get",
                "description": "Get a specific Gmail thread",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "thread_id": {
                            "type": "string",
                            "description": "ID of the thread to retrieve"
                        }
                    },
                    "required": ["thread_id"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_threads_trash",
                "description": "Move a Gmail thread to trash",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "thread_id": {
                            "type": "string",
                            "description": "ID of the thread to move to trash"
                        }
                    },
                    "required": ["thread_id"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_threads_untrash",
                "description": "Remove a Gmail thread from trash",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "thread_id": {
                            "type": "string",
                            "description": "ID of the thread to remove from trash"
                        }
                    },
                    "required": ["thread_id"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "tasks_tasklists_delete",
                "description": "Delete a task list",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tasklist_id": {
                            "type": "string",
                            "description": "ID of the task list to delete"
                        }
                    },
                    "required": ["tasklist_id"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "tasks_tasklists_get",
                "description": "Get a specific task list",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tasklist_id": {
                            "type": "string",
                            "description": "ID of the task list to retrieve"
                        }
                    },
                    "required": ["tasklist_id"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "tasks_tasklists_insert",
                "description": "Create a new task list",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Title of the new task list"
                        }
                    },
                    "required": ["title"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "tasks_tasklists_list",
                "description": "List all task lists",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "tasks_tasklists_patch",
                "description": "Update a task list's metadata",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tasklist_id": {
                            "type": "string",
                            "description": "ID of the task list to update"
                        },
                        "title": {
                            "type": "string",
                            "description": "New title for the task list"
                        }
                    },
                    "required": ["tasklist_id", "title"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "tasks_tasklists_update",
                "description": "Replace a task list's metadata",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tasklist_id": {
                            "type": "string",
                            "description": "ID of the task list to update"
                        },
                        "title": {
                            "type": "string",
                            "description": "New title for the task list"
                        }
                    },
                    "required": ["tasklist_id", "title"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "tasks_clear",
                "description": "Clear all completed tasks from a task list",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tasklist_id": {
                            "type": "string",
                            "description": "ID of the task list to clear"
                        }
                    },
                    "required": ["tasklist_id"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "tasks_delete",
                "description": "Delete a task",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tasklist_id": {
                            "type": "string",
                            "description": "ID of the task list containing the task"
                        },
                        "task_id": {
                            "type": "string",
                            "description": "ID of the task to delete"
                        }
                    },
                    "required": ["tasklist_id", "task_id"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "tasks_get",
                "description": "Get a specific task",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tasklist_id": {
                            "type": "string",
                            "description": "ID of the task list containing the task"
                        },
                        "task_id": {
                            "type": "string",
                            "description": "ID of the task to retrieve"
                        }
                    },
                    "required": ["tasklist_id", "task_id"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "tasks_insert",
                "description": "Create a new task",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tasklist_id": {
                            "type": "string",
                            "description": "ID of the task list"
                        },
                        "title": {
                            "type": "string",
                            "description": "Title of the task"
                        },
                        "notes": {
                            "type": "string",
                            "description": "Notes for the task"
                        },
                        "due": {
                            "type": "string",
                            "description": "Due date in RFC 3339 format"
                        }
                    },
                    "required": ["tasklist_id", "title"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "tasks_list",
                "description": "List tasks in a task list",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tasklist_id": {
                            "type": "string",
                            "description": "ID of the task list"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of tasks to return (default: 100)"
                        }
                    },
                    "required": ["tasklist_id"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "tasks_move",
                "description": "Move a task to a different position",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tasklist_id": {
                            "type": "string",
                            "description": "ID of the task list"
                        },
                        "task_id": {
                            "type": "string",
                            "description": "ID of the task to move"
                        },
                        "parent": {
                            "type": "string",
                            "description": "New parent task ID (optional)"
                        },
                        "previous": {
                            "type": "string",
                            "description": "Previous sibling task ID (optional)"
                        }
                    },
                    "required": ["tasklist_id", "task_id"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "tasks_patch",
                "description": "Update a task's metadata",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tasklist_id": {
                            "type": "string",
                            "description": "ID of the task list"
                        },
                        "task_id": {
                            "type": "string",
                            "description": "ID of the task to update"
                        },
                        "title": {
                            "type": "string",
                            "description": "New title for the task"
                        },
                        "notes": {
                            "type": "string",
                            "description": "New notes for the task"
                        },
                        "due": {
                            "type": "string",
                            "description": "New due date in RFC 3339 format"
                        }
                    },
                    "required": ["tasklist_id", "task_id"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "tasks_update",
                "description": "Replace a task's metadata",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tasklist_id": {
                            "type": "string",
                            "description": "ID of the task list"
                        },
                        "task_id": {
                            "type": "string",
                            "description": "ID of the task to update"
                        },
                        "title": {
                            "type": "string",
                            "description": "New title for the task"
                        },
                        "notes": {
                            "type": "string",
                            "description": "New notes for the task"
                        },
                        "due": {
                            "type": "string",
                            "description": "New due date in RFC 3339 format"
                        }
                    },
                    "required": ["tasklist_id", "task_id", "title"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calendar_colors_get",
                "description": "Get calendar color definitions",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calendar_events_list",
                "description": "List calendar events",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "calendar_id": {
                            "type": "string",
                            "description": "Calendar ID (default: 'primary')"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of events to return"
                        },
                        "time_min": {
                            "type": "string",
                            "description": "Start time in ISO format (optional)"
                        },
                        "time_max": {
                            "type": "string",
                            "description": "End time in ISO format (optional)"
                        }
                    },
                    "required": [],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calendar_events_quick_add",
                "description": "Quickly add an event from text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "calendar_id": {
                            "type": "string",
                            "description": "Calendar ID"
                        },
                        "text": {
                            "type": "string",
                            "description": "Text description of the event"
                        }
                    },
                    "required": ["calendar_id", "text"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calendar_freebusy_query",
                "description": "Query free/busy information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "time_min": {
                            "type": "string",
                            "description": "Start time in ISO format"
                        },
                        "time_max": {
                            "type": "string",
                            "description": "End time in ISO format"
                        },
                        "calendar_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of calendar IDs to query"
                        }
                    },
                    "required": ["time_min", "time_max"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_drafts_update",
                "description": "Update an existing draft",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "draft_id": {
                            "type": "string",
                            "description": "ID of the draft to update"
                        },
                        "to": {
                            "type": "string",
                            "description": "New recipient email address (optional)"
                        },
                        "subject": {
                            "type": "string",
                            "description": "New email subject (optional)"
                        },
                        "body": {
                            "type": "string",
                            "description": "New email body content (optional)"
                        }
                    },
                    "required": ["draft_id"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_labels_modify",
                "description": "Modify an existing label",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "label_id": {
                            "type": "string",
                            "description": "ID of the label to modify"
                        },
                        "name": {
                            "type": "string",
                            "description": "New name for the label (optional)"
                        },
                        "visibility": {
                            "type": "string",
                            "description": "New visibility setting (optional)",
                            "enum": ["labelShow", "labelHide", "labelShowIfUnread"]
                        }
                    },
                    "required": ["label_id"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_messages_create",
                "description": "Create a new message (as draft or sent)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": "string",
                            "description": "Recipient email address"
                        },
                        "subject": {
                            "type": "string",
                            "description": "Email subject"
                        },
                        "body": {
                            "type": "string",
                            "description": "Email body content"
                        },
                        "draft": {
                            "type": "boolean",
                            "description": "Whether to create as draft (true) or send immediately (false)"
                        }
                    },
                    "required": ["to", "subject", "body"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_messages_delete",
                "description": "Permanently delete a message",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message_id": {
                            "type": "string",
                            "description": "ID of the message to delete"
                        }
                    },
                    "required": ["message_id"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "gmail_messages_import",
                "description": "Import a raw message",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "raw_email": {
                            "type": "string",
                            "description": "Raw email content to import"
                        },
                        "labels": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Labels to apply to the imported message"
                        }
                    },
                    "required": ["raw_email"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calendar_events_instances",
                "description": "Get instances of a recurring event",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "calendar_id": {
                            "type": "string",
                            "description": "Calendar ID"
                        },
                        "event_id": {
                            "type": "string",
                            "description": "ID of the recurring event"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of instances to return"
                        }
                    },
                    "required": ["calendar_id", "event_id"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calendar_events_move",
                "description": "Move an event to a different calendar",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "calendar_id": {
                            "type": "string",
                            "description": "Source calendar ID"
                        },
                        "event_id": {
                            "type": "string",
                            "description": "ID of the event to move"
                        },
                        "destination_id": {
                            "type": "string",
                            "description": "Destination calendar ID"
                        }
                    },
                    "required": ["calendar_id", "event_id", "destination_id"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calendar_events_patch",
                "description": "Update specific fields of an event",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "calendar_id": {
                            "type": "string",
                            "description": "Calendar ID"
                        },
                        "event_id": {
                            "type": "string",
                            "description": "ID of the event to update"
                        },
                        "summary": {
                            "type": "string",
                            "description": "New event title (optional)"
                        },
                        "description": {
                            "type": "string",
                            "description": "New event description (optional)"
                        },
                        "start": {
                            "type": "object",
                            "description": "New start time (optional)"
                        },
                        "end": {
                            "type": "object",
                            "description": "New end time (optional)"
                        }
                    },
                    "required": ["calendar_id", "event_id"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calendar_events_update",
                "description": "Replace all fields of an event",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "calendar_id": {
                            "type": "string",
                            "description": "Calendar ID"
                        },
                        "event_id": {
                            "type": "string",
                            "description": "ID of the event to update"
                        },
                        "event_data": {
                            "type": "object",
                            "description": "Complete event data to replace with"
                        }
                    },
                    "required": ["calendar_id", "event_id", "event_data"],
                    "additionalProperties": False
                }
            }
        }
    ]