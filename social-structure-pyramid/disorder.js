const fallStageText = document.getElementById('fall-stage-text');
const fallVisual = document.getElementById('fall-visual');
const stageTabs = Array.from(document.querySelectorAll('.stage-tab'));

const stages = [
  { text: '从有序高处开始下坠', className: 'stage-0' },
  { text: '节奏开始被打乱', className: 'stage-1' },
  { text: '关系开始变得拥挤', className: 'stage-2' },
  { text: '判断开始偏向即时反应', className: 'stage-3' },
  { text: '最终落入丛林法则', className: 'stage-4' }
];

function applyStage(index) {
  const stage = stages[index];

  if (fallStageText) {
    fallStageText.textContent = stage.text;
  }

  if (fallVisual) {
    fallVisual.className = `fall-visual ${stage.className}`;
  }

  stageTabs.forEach((tab, tabIndex) => {
    tab.classList.toggle('is-active', tabIndex === index);
  });
}

if (fallStageText || fallVisual || stageTabs.length) {
  let currentStage = 0;
  applyStage(currentStage);

  window.setInterval(() => {
    currentStage = (currentStage + 1) % stages.length;
    applyStage(currentStage);
  }, 1800);
}
