async function updateStatus(){
  try{
    const r = await fetch('/api/status', {cache:'no-store'});
    const s = await r.json();
    document.getElementById('gesture').textContent = s.gesture;
    document.getElementById('confidence').textContent = (Number(s.confidence)*100).toFixed(1)+'%';
    document.getElementById('emotion').textContent = s.emotion;
    document.getElementById('response').textContent = s.response;
    document.getElementById('speaking').textContent = s.speaking ? '● Speaking...' : '● Ready';
  }catch(e){}
}
setInterval(updateStatus, 250);
updateStatus();
