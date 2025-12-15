let convId = null;

function el(q){ return document.querySelector(q); }

// ---------------------------
// Start a new scenario
// ---------------------------
async function startScenario(){
    const form = document.getElementById('scenario-form');
    const fd = new FormData(form);
    const scenario = {};
    for(const [k,v] of fd.entries()) scenario[k] = v;

    try{
        const res = await fetch('/api/start', {
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body: JSON.stringify({scenario})
        });
        const data = await res.json();
        convId = data.conv_id;
        el('#scenario-meta').textContent = data.scenario_meta;
        el('#messages').innerHTML = '';
        appendSystem('Scenario started. You are the BUYER. Type your offer or use the microphone.')
    } catch(e){
        appendSystem('Error starting scenario: ' + e);
    }
}

// ---------------------------
// Append messages
// ---------------------------
function appendMessage(role, text, intent){
    const m = document.createElement('div');
    m.className = 'msg ' + (role==='Buyer' ? 'user' : 'bot');
    m.innerHTML = `<div class="role">${role}</div><div class="text">${text}</div>`;
    if(intent){
        const it = document.createElement('div'); it.className='intent'; it.textContent = `Intent: ${intent}`;
        m.appendChild(it);
    }
    el('#messages').appendChild(m);
    el('#messages').scrollTop = el('#messages').scrollHeight;
}

function appendSystem(text){
    const m = document.createElement('div'); m.className='system'; m.textContent=text;
    el('#messages').appendChild(m);
}

// ---------------------------
// Send text message to server
// ---------------------------
async function sendMessage(){
    const input = el('#message-input');
    const txt = input.value.trim();
    if(!txt || !convId) return;
    appendMessage('Buyer', txt);
    input.value='';

    try{
        const res = await fetch('/api/message',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({conv_id: convId, message: txt})
        });
        const data = await res.json();
        if(data.error){ appendSystem(data.error); return; }
        appendMessage('Seller', data.seller_text, data.intent);
        // Play Piper audio if returned
        if(data.audio_b64){
            const audio = new Audio('data:audio/wav;base64,' + data.audio_b64);
            audio.play().catch(()=>{});
        } else if(window.ttsEnabled){
            speakText(data.seller_text);
        }
    } catch(e){
        appendSystem('Server error: ' + e);
    }
}

// ---------------------------
// Client-side Speech-to-Text (Web Speech API)
// ---------------------------
let recognition = null;
const micBtn = document.getElementById('mic-btn');

if('webkitSpeechRecognition' in window || 'SpeechRecognition' in window){
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SR();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => { micBtn.classList.add('listening'); micBtn.textContent = '●'; };
    recognition.onend = () => { micBtn.classList.remove('listening'); micBtn.textContent = '🎤'; };
    recognition.onerror = (e) => { console.error('Speech error', e); appendSystem('Speech recognition error: ' + e.error); };

    recognition.onresult = (ev) => {
        const transcript = Array.from(ev.results).map(r=>r[0].transcript).join('');
        if(!transcript) return;
        appendMessage('Buyer', transcript);
        // Send recognized text to server
        fetch('/api/message',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({conv_id: convId, message: transcript})
        }).then(r=>r.json()).then(data=>{
            if(data.error) return appendSystem(data.error);
            appendMessage('Seller', data.seller_text, data.intent);
            if(data.audio_b64){
                const audio = new Audio('data:audio/wav;base64,' + data.audio_b64);
                audio.play().catch(()=>{});
            } else if(window.ttsEnabled){
                speakText(data.seller_text);
            }
        }).catch(err=>appendSystem('Server error: '+err));
    };

    micBtn.addEventListener('click', ()=>{
        if(!convId){ appendSystem('Start a scenario first.'); return; }
        try{ recognition.start(); } catch(e){ console.warn(e); }
    });
} else {
    micBtn.disabled = true;
    micBtn.title = 'Browser does not support Web Speech API';
}

// ---------------------------
// Text-to-Speech: browser fallback
// ---------------------------
window.ttsEnabled = true;
function speakText(text){
    if(!('speechSynthesis' in window)) return;
    try{
        const u = new SpeechSynthesisUtterance(text);
        u.lang = 'en-US';
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(u);
    } catch(e){ console.warn('TTS error', e); }
}
// ---------------------------
// End scenario & request grading
// ---------------------------
async function endAndGrade(){
    if(!convId){
        appendSystem("Start a scenario first.");
        return;
    }

    appendSystem("Grading your negotiation…");

    try{
        const res = await fetch('/api/grade', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ conv_id: convId })
        });

        const data = await res.json();
        if(data.error){
            appendSystem("Grading error: " + data.error);
            return;
        }

        appendMessage('System', "📊 Negotiation Grade");
        appendMessage('System', data.summary);
        appendMessage('System', "Score: " + data.score + "/100");
    } catch(e){
        appendSystem("Server error: " + e);
    }
}

document.getElementById('end-btn').addEventListener('click', endAndGrade);
document.getElementById('start-btn').addEventListener('click', startScenario);
document.getElementById('send-btn').addEventListener('click', sendMessage);
document.getElementById('message-input').addEventListener('keydown', (e)=>{ if(e.key==='Enter') sendMessage(); });

// TTS toggle button
const ttsBtn = document.getElementById('tts-btn');
if(ttsBtn){
    const update = ()=>{ ttsBtn.style.opacity = window.ttsEnabled ? '1' : '0.45'; };
    ttsBtn.addEventListener('click', ()=>{ window.ttsEnabled = !window.ttsEnabled; update(); });
    update();
}
