import threading
import time
import PySimpleGUI as sg
from it_works import (
    default_scenario,
    record_audio,
    speech_to_text,
    generate_reply,
    run_guardrail,
    speak,
    initialize_models,
    get_init_status,
    is_initialized,
)

# Simple GUI to drive the negotiation script

sg.theme('DarkTeal9')

layout_setup = [
    [sg.Text('Negotiation Scenario', font=('Helvetica', 16))],
    [sg.Text('Category', size=(15,1)), sg.Input(key='-CATEGORY-', default_text='electronics')],
    [sg.Text('Item name', size=(15,1)), sg.Input(key='-ITEM-', default_text='iPhone 12 (128GB)')],
    [sg.Text('Description', size=(15,1)), sg.Input(key='-DESC-', default_text='Used, good battery')],
    [sg.Text('List price', size=(15,1)), sg.Input(key='-LIST-', default_text='500')],
    [sg.Text('Seller target price', size=(15,1)), sg.Input(key='-SELLER_T-', default_text='420')],
    [sg.Text('Seller bottomline', size=(15,1)), sg.Input(key='-SELLER_B-', default_text='380')],
    [sg.Text('Buyer target price', size=(15,1)), sg.Input(key='-BUYER_T-', default_text='350')],
    [sg.Text('Buyer bottomline', size=(15,1)), sg.Input(key='-BUYER_B-', default_text='400')],
    [sg.Button('Start Conversation', key='-START-', button_color=('white','green'), disabled=True), sg.Button('Quit')],
    [sg.Text('Status:'), sg.Text('Initializing...', key='-STATUS-', size=(40,1))]
]

layout_conv = [
    [sg.Text('Conversation', font=('Helvetica', 16))],
    [sg.Multiline('', size=(60,12), key='-TRANSCRIPT-', disabled=True, autoscroll=True)],
    [sg.Text('Last Intent:'), sg.Text('', key='-INTENT-', size=(30,1), text_color='yellow')],
    [sg.Button('Record Offer (5s)', key='-RECORD-', button_color=('white','blue')), sg.Button('End Conversation', key='-END-')]
]

layout = [
    [sg.Column(layout_setup, key='-SETUP-'), sg.VerticalSeparator(), sg.Column(layout_conv, key='-CONV-', visible=False)]
]

window = sg.Window('Negotiation Assistant', layout, finalize=True, element_justification='left')

conversation_running = False
history = []
scenario = default_scenario()

# Helper to append text to transcript
def append_transcript(text):
    cur = window['-TRANSCRIPT-'].get()
    new = cur + text + '\n'
    window['-TRANSCRIPT-'].update(new)

# Background worker for handling a single record -> process -> speak cycle
def handle_record():
    global history, scenario
    try:
        append_transcript('[SYSTEM] Recording 5 seconds...')
        audio = record_audio(duration=5)
        buyer_msg = speech_to_text(audio).strip()
        append_transcript(f'[BUYER] {buyer_msg}')

        seller_raw, intent = generate_reply(None, None, scenario, history, buyer_msg)
        seller_final, intent_final = run_guardrail(buyer_msg, seller_raw, intent, scenario)

        append_transcript(f'[SELLER] {seller_final}   (intent: {intent_final})')
        window['-INTENT-'].update(intent_final)

        # Speak in a separate thread to avoid blocking UI
        t = threading.Thread(target=speak, args=(seller_final,), daemon=True)
        t.start()

        history.append(('Buyer', buyer_msg))
        history.append(('Seller', seller_final))
    except Exception as e:
        append_transcript(f'[ERROR] {e}')

# Main event loop
# Start initialization in background thread and stream status updates to the UI
def init_worker():
    # Start initialization in a separate thread so we can poll status
    def _run_init():
        try:
            initialize_models(use_cuda=False)
        except Exception:
            pass

    t = threading.Thread(target=_run_init, daemon=True)
    t.start()

    # Poll status until ready or error
    while True:
        status = get_init_status()
        window.write_event_value('-INIT_STATUS-', status)
        if status.get('ready') or status.get('step') == 'error':
            break
        time.sleep(0.5)

threading.Thread(target=init_worker, daemon=True).start()

while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Quit'):
        break

    if event == '-INIT_STATUS-':
        st = values[event]
        msg = st.get('message') if isinstance(st, dict) else str(st)
        ready = st.get('ready', False) if isinstance(st, dict) else False
        window['-STATUS-'].update(msg)
        if ready:
            window['-START-'].update(disabled=False)
        continue

    if event == '-START-':
        # Build scenario from inputs
        try:
            scenario = {
                'category': values['-CATEGORY-'],
                'item_name': values['-ITEM-'],
                'item_description': values['-DESC-'],
                'list_price': float(values['-LIST-']),
                'seller_target_price': float(values['-SELLER_T-']),
                'seller_bottomline': float(values['-SELLER_B-']),
                'buyer_target_price': float(values['-BUYER_T-']),
                'buyer_bottomline': float(values['-BUYER_B-'])
            }
        except Exception:
            sg.popup('Please enter valid numeric prices.')
            continue

        window['-SETUP-'].update(visible=False)
        window['-CONV-'].update(visible=True)
        append_transcript('[SYSTEM] Conversation started. Press Record Offer to speak.')
        conversation_running = True

    if event == '-RECORD-' and conversation_running:
        # run recording in background thread to keep UI responsive
        threading.Thread(target=handle_record, daemon=True).start()

    if event == '-END-' and conversation_running:
        # Produce final summary
        append_transcript('[SYSTEM] Conversation ended.')
        # Determine final intent summary (last non-unknown intent)
        intents = []
        # intents can be read from transcript lines showing (intent: ...)
        cur = window['-TRANSCRIPT-'].get()
        for line in cur.splitlines():
            if '(intent:' in line:
                try:
                    part = line.split('(intent:')[-1]
                    intents.append(part.strip(') ').strip())
                except:
                    pass
        final_intent = intents[-1] if intents else 'unknown'
        sg.popup('Conversation Summary', f'Number of turns: {len(history)//2}', f'Final intent: {final_intent}')
        conversation_running = False
        # allow the user to go back to setup
        window['-SETUP-'].update(visible=True)
        window['-CONV-'].update(visible=False)
        history = []
        window['-TRANSCRIPT-'].update('')
        window['-INTENT-'].update('')

window.close()
