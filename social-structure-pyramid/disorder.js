const fallStageText = document.getElementById('fall-stage-text');
const fallVisual = document.getElementById('fall-visual');
const stageTabs = Array.from(document.querySelectorAll('.stage-tab'));

const stages = [
  { text: '从有序高处开始下坠', className: 'stage-0', duration: 1800 },
  { text: '节奏逐步失去稳定', className: 'stage-1', duration: 1800 },
  { text: '关系边界开始松动', className: 'stage-2', duration: 1800 },
  { text: '判断越来越偏向即时反应', className: 'stage-3', duration: 1800 },
  { text: '穿透跌落之后，将面对丛林法则般的生存环境', className: 'stage-4', duration: 5000 }
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

  const scheduleNextStage = () => {
    applyStage(currentStage);

    window.setTimeout(() => {
      currentStage = (currentStage + 1) % stages.length;
      scheduleNextStage();
    }, stages[currentStage].duration);
  };

  scheduleNextStage();
}
