import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js";
import Meyda from "https://cdn.jsdelivr.net/npm/meyda@5.4.0/dist/web/meyda.min.js"; // Meyda import 추가

const SAMPLE_RATE = 16000; // 오디오 샘플링 레이트
const MFCC_DIM = 13;       // MFCC 계수 개수 (학습 모델과 일치: 13)

// 학습 시 torchaudio.transforms.MFCC의 n_fft 및 hop_length와 일치시켜야 함
// torchaudio 기본값: n_fft=400, hop_length=160
const FFT_SIZE = 400;      // Meyda의 bufferSize에 해당
const HOP_SIZE = 160;      // Meyda의 hopSize에 해당

// 1초 오디오 (16000 샘플)가 MFCC로 변환될 때의 프레임 수 계산
// (SAMPLE_RATE - FFT_SIZE) / HOP_SIZE + 1 = (16000 - 400) / 160 + 1 = 98.5 + 1 = 99.5
// 모델이 1초 오디오를 처리한다면 약 99~100 프레임이 나와야 합니다.
// 학습 시 정확한 프레임 수를 확인하여 이 값을 설정해야 합니다.
const MFCC_FRAMES = 100; // 49 -> 100으로 변경 (torchaudio 기본값 기준)

const WAKEWORD_THRESHOLD = 0.7; // 웨이크워드 탐지 임계값

const statusEl = document.getElementById("status"); // HTML에 status 엘리먼트가 있다고 가정

let session = null;
let audioContext = null;
let analyzer = null;
let mfccBuffer = []; // MFCC 프레임을 저장할 버퍼 (audioBuffer -> mfccBuffer로 이름 변경)
let isProcessing = false; // 예측 중복 실행 방지 플래그

async function predictWakeword(mfccArray) {
  // mfccArray: Float32Array (MFCC_DIM * MFCC_FRAMES)
  // ONNX 입력은 (1, 1, MFCC_DIM, MFCC_FRAMES)

  // ONNX Tensor 생성 (shape: [Batch, Channel, Height, Width])
  const inputTensor = new ort.Tensor("float32", mfccArray, [1, 1, MFCC_DIM, MFCC_FRAMES]);
  
  // 모델 추론 실행
  const results = await session.run({ input: inputTensor });
  // ONNX 모델의 첫 번째 출력 (로짓 값)을 가져옴
  const outputData = results[session.outputNames[0]].data; 

  // 소프트맥스 직접 계산
  const exp0 = Math.exp(outputData[0]); // negative class logit
  const exp1 = Math.exp(outputData[1]); // positive (wakeword) class logit
  const probWakeword = exp1 / (exp0 + exp1);

  return probWakeword;
}

function convertFramesToTensor(mfccFrames) {
  // mfccFrames: Array of MFCC frame arrays (each length MFCC_DIM)
  // 프레임이 MFCC_FRAMES가 안되면 0으로 패딩하여 고정된 크기 반환

  const frames = new Float32Array(MFCC_DIM * MFCC_FRAMES);
  for (let i = 0; i < MFCC_FRAMES; i++) {
    if (i < mfccFrames.length) {
      // 각 MFCC 프레임(배열)을 frames 배열에 평탄화하여 복사
      for (let m = 0; m < MFCC_DIM; m++) {
        frames[i * MFCC_DIM + m] = mfccFrames[i][m];
      }
    } else {
      // 부족한 프레임은 0으로 패딩 (Float32Array는 기본적으로 0으로 초기화되므로 명시적 패딩 생략 가능)
      // 하지만 명확성을 위해 코드는 유지
      for (let m = 0; m < MFCC_DIM; m++) {
        frames[i * MFCC_DIM + m] = 0;
      }
    }
  }
  return frames;
}

async function startAudioProcessing() {
  audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const source = audioContext.createMediaStreamSource(stream);

  // Meyda analyzer 설정 (학습 시 torchaudio 파라미터와 일치하도록 설정)
  analyzer = Meyda.createMeydaAnalyzer({
    audioContext,
    source,
    bufferSize: FFT_SIZE, // torchaudio의 n_fft와 일치
    hopSize: HOP_SIZE,   // torchaudio의 hop_length와 일치
    featureExtractors: ["mfcc"],
    numberOfMFCCCoefficients: MFCC_DIM, // MFCC 계수 개수 설정 (13)
    callback: async (features) => {
      if (!features || !features.mfcc) return;
      
      // 현재 추출된 MFCC 프레임을 버퍼에 추가
      mfccBuffer.push(features.mfcc);

      // 버퍼에 충분한 프레임이 모이고, 현재 예측 중이 아니라면 예측 시작
      // MFCC_FRAMES * 2는 버퍼가 너무 커지는 것을 방지하기 위한 임시 값.
      // 실제 사용 시에는 이 버퍼 관리 로직을 더 정교하게 만들 수 있습니다.
      if (mfccBuffer.length >= MFCC_FRAMES && !isProcessing) {
        isProcessing = true; // 예측 중임을 표시

        // 가장 최근 MFCC_FRAMES 수만큼의 프레임만 사용
        const mfccFramesToProcess = mfccBuffer.slice(-MFCC_FRAMES);
        const mfccTensorData = convertFramesToTensor(mfccFramesToProcess);

        try {
          const prob = await predictWakeword(mfccTensorData);
          if (prob >= WAKEWORD_THRESHOLD) {
            statusEl.innerText = `Wake Word Detected (확률: ${(prob * 100).toFixed(1)}%)`;
            console.log(`Wake Word Detected (확률: ${(prob * 100).toFixed(1)}%)`);
          } else {
            statusEl.innerText = `No Wake Word (확률: ${(prob * 100).toFixed(1)}%)`;
            // console.log(`No Wake Word (확률: ${(prob * 100).toFixed(1)}%)`); // 너무 많이 찍힐 경우 주석 처리
          }
        } catch (e) {
          console.error("웨이크워드 예측 중 오류 발생:", e);
          statusEl.innerText = "오류 발생!";
        } finally {
          isProcessing = false; // 예측 완료 후 플래그 해제
          // 버퍼 크기 관리: 너무 커지면 오래된 데이터 제거
          if (mfccBuffer.length > MFCC_FRAMES * 2) { 
              mfccBuffer = mfccBuffer.slice(-MFCC_FRAMES);
          }
        }
      }
    }
  });

  analyzer.start(); // Meyda Analyzer 시작
  statusEl.innerText = "모델 로딩 완료. 음성 인식 시작 중...";
}

async function init() {
  statusEl.innerText = "모델 로딩 중...";
  try {
    // 전역 session 변수에 할당 (const 제거)
    session = await ort.InferenceSession.create("http://localhost:4000/models/wakeword_crnn.onnx");
    statusEl.innerText = "모델 로딩 완료. 마이크 접근 대기 중...";
    await startAudioProcessing();
  } catch (e) {
    statusEl.innerText = "모델 로딩 실패!";
    console.error("ONNX 모델 로딩 오류:", e);
  }
}

init();