<script lang="ts">
import { defineComponent, type UnwrapRef, reactive, markRaw, toRaw } from 'vue';
import { fabric } from 'fabric';
import { PlusSquareOutlined, UploadOutlined } from '@ant-design/icons-vue';
import OpenposeObjectPanel from './components/OpenposeObjectPanel.vue';
import {
  OpenposePerson,
  OpenposeBody,
  OpenposeHand,
  OpenposeFace,
  OpenposeKeypoint2D,
  OpenposeObject,
  type IOpenposeJson,
  OpenposeBodyPart,
  OpenposeAnimal,
} from './Openpose';
import type { UploadFile } from 'ant-design-vue';
import _ from 'lodash';
import CryptoJS from 'crypto-js';

interface LockableUploadFile extends UploadFile {
  locked: boolean;
  scale: number;
  upload_url?: String;
};

interface AppData {
  canvasHeight: number;
  canvasWidth: number;

  pos: string;
  neg: string;
  num_inference_steps: number;
  seed: number;
  img_p_scale: number;
  ctrl_scale: number;
  cfg: number;

  imageURL: string;
  personName: string;
  hideInvisibleKeypoints: boolean;
  people: Map<number, OpenposePerson>;
  canvas: HTMLCanvasElement | null;
  openposeCanvas: fabric.Rect;
  keypointMap: Map<number, UnwrapRef<OpenposeKeypoint2D>>;
  uploadedImageList: Array<any>;
  temp_img: HTMLImageElement | null;
  canvasImageMap: Map<string, fabric.Image>;
  activePersonId: number | undefined;
  activeBodyPart: string | undefined;
  modalId: string | undefined;
  color: string;

  conditionURL: string;

  referenceURL: string;

  canvasVisible: boolean;
  maskContext: CanvasRenderingContext2D | null;
  maskDrawingCanvas: HTMLCanvasElement | null;
  maskDrawing: boolean;
  maskURL: string;

  latentVisible: boolean;
  latentContext: CanvasRenderingContext2D | null;
  latentDrawingCanvas: HTMLCanvasElement | null;
  latentDrawing: boolean,
  maskURL: string;

  reference2Visible: boolean
  referenceURL2: string;

  canvasVisible2: boolean;
  maskContext2: CanvasRenderingContext2D | null;
  maskDrawingCanvas2: HTMLCanvasElement | null;
  maskDrawing2: boolean;
  maskURL2: string;

  latentVisible2: boolean;
  latentContext2: CanvasRenderingContext2D | null;
  latentDrawingCanvas2: HTMLCanvasElement | null;
  latentDrawing2: boolean,
  latentURL2: string;
}

/**
 * The frame message from the main frame (ControlNet).
 */
interface IncomingFrameMessage {
  modalId: string;
  imageURL?: string;
  poseURL?: string;
  poses?: IOpenposeJson | IOpenposeJson[];
};

interface OutgoingFrameMessage {
  modalId: string;
  poseURL: string;
  poses: IOpenposeJson;
};

const default_body_keypoints: [number, number, number][] = [
  [241, 77], [241, 120], [191, 118], [177, 183],
  [163, 252], [298, 118], [317, 182], [332, 245],
  [225, 241], [213, 359], [215, 454], [270, 240],
  [282, 360], [286, 456], [232, 59], [253, 60],
  [225, 70], [260, 72]
].map(loc => [loc[0], loc[1], 1.0]);

const default_left_hand_keypoints: [number, number, number][] = [
  [
    72.0,
    138.6749968987715,
    1
  ],
  [
    50.00001525878906,
    126.6749968987715,
    1
  ],
  [
    26.00001525878906,
    109.6749968987715,
    1
  ],
  [
    15.0,
    89.6749968987715,
    1
  ],
  [
    0.001,
    74.6749968987715,
    1
  ],
  [
    46.88441843878036,
    68.06475585206891,
    1
  ],
  [
    44.700867221123644,
    41.31155552497435,
    1
  ],
  [
    42.99998474121094,
    22.714115786649984,
    1
  ],
  [
    42.00298865302648,
    7.0136598257858935,
    1
  ],
  [
    66.64955876004365,
    63.25333859753661,
    1
  ],
  [
    65.00001525878906,
    40.6749968987715,
    1
  ],
  [
    65.94870679770906,
    22.18139755296059,
    1
  ],
  [
    65.38117571016846,
    0.001,
    1
  ],
  [
    82.28699289317831,
    68.16587713769567,
    1
  ],
  [
    84.50425981167291,
    45.869876375420176,
    1
  ],
  [
    85.4529971269601,
    27.623076702514766,
    1
  ],
  [
    85.02609712479875,
    9.122938432072758,
    1
  ],
  [
    98.0260818660098,
    76.01283565581218,
    1
  ],
  [
    103.05127794350199,
    60.272436637095865,
    1
  ],
  [
    107.35044123995635,
    46.921796571676936,
    1
  ],
  [
    110.70086722112376,
    31.519236310001304,
    1
  ]
];

const default_right_hand_keypoints: [number, number, number][] = [
  [
    37.00000762939453,
    140.03029482565358,
    1
  ],
  [
    59.000003814697266,
    132.03029482565358,
    1
  ],
  [
    83.00000381469727,
    117.0302948256536,
    1
  ],
  [
    99.99999618530273,
    99.0302948256536,
    1
  ],
  [
    117.99999618530273,
    88.0302948256536,
    1
  ],
  [
    68.60503479194651,
    69.66265371825791,
    1
  ],
  [
    72.0000114440918,
    51.0302948256536,
    1
  ],
  [
    75.99999618530273,
    34.0302948256536,
    1
  ],
  [
    80.00000381469727,
    17.0302948256536,
    1
  ],
  [
    47.878141976595,
    66.66265371825791,
    1
  ],
  [
    49.424375419378265,
    45.4861751947855,
    1
  ],
  [
    51.0,
    21.0302948256536,
    1
  ],
  [
    54.0,
    0.001,
    1
  ],
  [
    29.575632210016238,
    70.81461384130189,
    1
  ],
  [
    30.0,
    44.574414456521666,
    1
  ],
  [
    30.848747024059264,
    26.150394518043655,
    1
  ],
  [
    34.57563602471356,
    7.270494210433782,
    1
  ],
  [
    11.000003814697266,
    78.99843439499973,
    1
  ],
  [
    8.000003814697266,
    61.0302948256536,
    1
  ],
  [
    4.000007629394531,
    48.0302948256536,
    1
  ],
  [
    0.001,
    34.0302948256536,
    1
  ]
];

const default_face_keypoints: [number, number, number][] = [];

// identity_metrics * point == point.
const IDENTITY_MATRIX = [1, 0, 0, 1, 0, 0];

function parseDataURLtoJSON(dataURL: string): any {
  const data = dataURL.split(',')[1]; // Extract the data portion
  const decodedData = atob(data); // Decode the data
  const json = JSON.parse(decodedData); // Parse the decoded data as JSON
  return json;
}

function serializeJSONtoDataURL(data: any): string {
  return "data:application/json;base64," + btoa(JSON.stringify(data));
}

async function calculateHash(s: string): Promise<string> {
  return CryptoJS.SHA256(s).toString()
}

function getImageDimensionsFromDataURL(dataURL: string): Promise<[number, number]> {
  return new Promise((resolve, reject) => {
    const img = new Image();

    img.onload = function () {
      const width = img.naturalWidth;
      const height = img.naturalHeight;
      resolve([width, height]);
    };

    img.onerror = function () {
      reject(new Error('Failed to load image.'));
    };

    img.src = dataURL;
  });
}

export default defineComponent({
  data(): AppData {
    return {
      canvasHeight: 512,
      canvasWidth: 512,
      conditionURL: '',

      pos: "girl, beach, best quality, masterpiece",
      neg: "ugly, monochrome",
      num_inference_steps: 12,
      img_p_scale: 0.7,
      ctrl_scale: 0.8,
      cfg: 5,
      seed: 42,

      imageURL: '',
      personName: '',
      hideInvisibleKeypoints: false,
      people: new Map<number, OpenposePerson>(),
      canvas: null,
      openposeCanvas: new fabric.Rect({
        fill: "#000",
        selectable: false,
        evented: false,
      }),

      keypointMap: new Map<number, UnwrapRef<OpenposeKeypoint2D>>(),
      uploadedImageList: [],
      temp_img: null,
      canvasImageMap: new Map<string, fabric.Image>(),
      activePersonId: undefined,
      activeBodyPart: undefined,
      modalId: undefined,
      color: 'white',

      referenceURL: '',

      maskContext: null,
      canvasVisible: true,
      maskDrawingCanvas: null,
      maskDrawing: false,
      maskURL: '',

      latentContext: null,
      latentVisible: false,
      latentDrawingCanvas: null,
      latentDrawing: false,
      latentURL: '',

      reference2Visible: false,
      referenceURL2: '',

      canvasVisible2: true,
      maskContext2: null,
      maskDrawingCanvas2: null,
      maskDrawing2: false,
      maskURL2: '',

      latentVisible2: true,
      latentContext2: null,
      latentDrawingCanvas2: null,
      latentDrawing2: false,
      latentURL2: '',
    };
  },
  setup() {
    return { OpenposeBodyPart };
  },
  mounted() {
    const canvas_draw = this.$refs.maskDrawingCanvas as HTMLCanvasElement;
    this.maskDrawingCanvas = canvas_draw;
    this.maskContext = canvas_draw.getContext('2d');

    const canvas_draw_latent = this.$refs.latentDrawingCanvas as HTMLCanvasElement;
    this.latentDrawingCanvas = canvas_draw_latent;
    this.latentContext = canvas_draw_latent.getContext('2d');

    const canvas_draw2 = this.$refs.maskDrawingCanvas2 as HTMLCanvasElement;
    this.maskDrawingCanvas2 = canvas_draw2;
    this.maskContext2 = canvas_draw2.getContext('2d');

    const canvas_draw_latent2 = this.$refs.latentDrawingCanvas2 as HTMLCanvasElement;
    this.latentDrawingCanvas2 = canvas_draw_latent2;
    this.latentContext2 = canvas_draw_latent2.getContext('2d');


    this.$nextTick(() => {
      this.canvas = markRaw(new fabric.Canvas(<HTMLCanvasElement>this.$refs.editorCanvas));

      this.resizeHTMLCanvas();
      this.canvas.add(this.openposeCanvas);
      // The openpose canvas should be at last layer.
      this.canvas.moveTo(this.openposeCanvas, 0);
      this.resizeOpenposeCanvas(this.canvasWidth, this.canvasHeight);

      // By default have a example person.
      this.addDefaultPerson();

      // eslint-disable-next-line
      const selectionHandler = (event: fabric.IEvent<MouseEvent>) => {
        if (event.selected) {
          event.selected
            .filter(o => o instanceof OpenposeKeypoint2D)
            .forEach(p => {
              const proxy = this.getKeypointProxy(p as OpenposeKeypoint2D);
              if (event.selected!.length > 1)
                proxy.selected_in_group = true;
              proxy.selected = true;
            });
        }

        if (event.deselected) {
          event.deselected
            .filter(o => o instanceof OpenposeKeypoint2D)
            .forEach(p => {
              const proxy = this.getKeypointProxy(p as OpenposeKeypoint2D);
              if (event.deselected!.length > 1)
                proxy.selected_in_group = false;
              proxy.selected = false;
            });
        }
      }

      const keypointMoveHandler = (event: fabric.IEvent<MouseEvent>) => {
        if (event.target === undefined)
          return;

        const target = event.target;
        if (target instanceof fabric.ActiveSelection) {
          // Group of points movement.
          const t = target.calcTransformMatrix();
          target.forEachObject(obj => {
            if (obj instanceof OpenposeKeypoint2D) {
              obj.updateConnections(t);
            }
          });
        } else if (target instanceof OpenposeKeypoint2D) {
          // Single keypoint movement.
          target.updateConnections(IDENTITY_MATRIX);
          this.updateKeypointProxy(target);
        }
        this.canvas?.renderAll();
      };

      const hideKeypointHandler = (event: fabric.IEvent<MouseEvent>) => {
        // Only handles right click events.
        if (event.button !== 3) return;
        if (!(event.target instanceof OpenposeKeypoint2D)) return;
        event.target._visible = false;

        this.canvas?.renderAll();
      };

      this.canvas.on('object:moving', keypointMoveHandler);
      this.canvas.on('object:scaling', keypointMoveHandler);
      this.canvas.on('object:rotating', keypointMoveHandler);
      this.canvas.on('selection:created', selectionHandler);
      this.canvas.on('selection:cleared', selectionHandler);
      this.canvas.on('selection:updated', selectionHandler);
      this.canvas.on('mouse:down', hideKeypointHandler);

      // Zoom handler.
      this.canvas.on('mouse:wheel', (opt: fabric.IEvent<WheelEvent>) => {
        const delta = opt.e.deltaY;
        let zoom = this.canvas!.getZoom();
        zoom *= 0.999 ** delta;
        if (zoom > 20) zoom = 20;
        if (zoom < 0.01) zoom = 0.01;
        this.canvas!.zoomToPoint({ x: opt.e.offsetX, y: opt.e.offsetY } as fabric.Point, zoom);
        opt.e.preventDefault();
        opt.e.stopPropagation();
      });

      // Panning handler.
      let panning = false;
      let panningEnabled = false;
      // Enable panning by press SPACE or F because some users report SPACE
      // scrolls the iframe despite `e.preventDefault` is called. Issue #7.
      // Add keydown event to document
      document.addEventListener('keydown', (e) => {
        if (e.key === '=') {
          panningEnabled = true;
          this.canvas!.selection = false;
          // Prevent default behaviour of Space which is scroll the page down.
          e.preventDefault();
        }
      });

      // Add keyup event to document
      document.addEventListener('keyup', (e) => {
        if (e.key === '=') {
          panningEnabled = false;
          this.canvas!.selection = true;
        }
      });

      // Attach the mouse down event to start panning
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      this.canvas.on('mouse:down', (opt: fabric.IEvent) => {
        if (panningEnabled) {
          panning = true;
        }
      });

      // Attach the mouse move event for panning
      this.canvas.on('mouse:move', (opt: fabric.IEvent<MouseEvent>) => {
        if (panning && opt && opt.e) {
          const delta = new fabric.Point(opt.e.movementX, opt.e.movementY);
          this.canvas!.relativePan(delta);
        }
      });

      // Attach the mouse up event to stop panning
      this.canvas.on('mouse:up', () => {
        panning = false;
      });

      // Handle incoming frame message.
      window.addEventListener('message', async (event) => {
        const message = event.data as IncomingFrameMessage;
        if (message.modalId === undefined) {
          console.debug(`Unrecognized frame message received: ${JSON.stringify(message)}.`);
          return;
        }
        await this.waitWindowVisible();
        this.resizeHTMLCanvas();
        this.loadCanvasFromFrameMessage(message);
      });

      // Inform the parent frame that iframe is ready to receive message.
      if (window.self != window.top) {
        window.parent.postMessage({
          ready: true
        }, '*');
      }
    });
    window.addEventListener('keydown', this.handleKeyDown);
  },
  beforeUnmount() {
    window.removeEventListener('keydown', this.handleKeyDown);
  },
  methods: {
    startDrawingOnMaskCanvas(event: MouseEvent, target: String): void {
      console.log(target)
      if (target == "mask") {
        this.maskDrawing = true;
        if (this.maskContext) {
          this.maskContext.beginPath();
          this.maskContext.moveTo(event.offsetX, event.offsetY);
        }
      } else if (target == "latent") {
        this.latentDrawing = true;
        if (this.latentContext) {
          this.latentContext.beginPath();
          this.latentContext.moveTo(event.offsetX, event.offsetY);
        }
      } else if (target == "mask2") {
        this.maskDrawing2 = true;
        if (this.maskContext2) {
          this.maskContext2.beginPath();
          this.maskContext2.moveTo(event.offsetX, event.offsetY);
        }
      } else if (target == "latent2") {
        this.latentDrawing2 = true;
        if (this.latentContext2) {
          this.latentContext2.beginPath();
          this.latentContext2.moveTo(event.offsetX, event.offsetY);
        }
      }
    },
    drawOnMaskCanvas(event: MouseEvent, target: String): void {
      if (target == "mask") {
        if (!this.maskDrawing || !this.maskContext) return;
        this.maskContext.lineTo(event.offsetX, event.offsetY);
        this.maskContext.strokeStyle = this.color;
        this.maskContext.lineWidth = 50;
        this.maskContext.stroke();
      } else if (target == "latent") {
        if (!this.latentDrawing || !this.latentContext) return;
        this.latentContext.lineTo(event.offsetX, event.offsetY);
        this.latentContext.strokeStyle = this.color;
        this.latentContext.lineWidth = 50;
        this.latentContext.stroke();
      } else if (target == "mask2") {
        if (!this.maskDrawing2 || !this.maskContext2) return;
        this.maskContext2.lineTo(event.offsetX, event.offsetY);
        this.maskContext2.strokeStyle = this.color;
        this.maskContext2.lineWidth = 50;
        this.maskContext2.stroke();
      } else if (target == "latent2") {
        if (!this.latentDrawing2 || !this.latentContext2) return;
        this.latentContext2.lineTo(event.offsetX, event.offsetY);
        this.latentContext2.strokeStyle = this.color;
        this.latentContext2.lineWidth = 50;
        this.latentContext2.stroke();
      }
    },
    stopDrawingMaskOnCanvas(target: String): void {
      if (target == "mask") {
        this.maskDrawing = false;
        if (this.maskDrawingCanvas) {
          this.maskURL = this.maskDrawingCanvas.toDataURL('image/png');
        }
      } else if (target == "latent") {
        this.latentDrawing = false;
        if (this.latentDrawingCanvas) {
          this.latentURL = this.latentDrawingCanvas.toDataURL('image/png');
        }
      } else if (target == "mask2") {
        this.maskDrawing2 = false;
        if (this.maskDrawingCanvas2) {
          this.maskURL2 = this.maskDrawingCanvas2.toDataURL('image/png');
        }
      } else if (target == "latent2") {
        this.latentDrawing2 = false;
        if (this.latentDrawingCanvas2) {
          this.latentURL2 = this.latentDrawingCanvas2.toDataURL('image/png');
        }
      }


    },

    clearDrawingCanvas(target: String): void {
      if (target == "mask") {
        if (this.maskContext && this.maskDrawingCanvas) {
          this.maskContext.clearRect(0, 0, this.maskDrawingCanvas.width, this.maskDrawingCanvas.height);
        }
        this.maskURL = '';
      } else if (target == "latent") {
        if (this.latentContext && this.latentDrawingCanvas) {
          this.latentContext.clearRect(0, 0, this.latentDrawingCanvas.width, this.latentDrawingCanvas.height);
        }
        this.latentURL = '';
      } else if (target == "mask2") {
        if (this.maskContext2 && this.maskDrawingCanvas2) {
          this.maskContext2.clearRect(0, 0, this.maskDrawingCanvas2.width, this.maskDrawingCanvas2.height);
        }
        this.maskURL2 = '';
      } else if (target == "latent2") {
        if (this.latentContext2 && this.latentDrawingCanvas2) {
          this.latentContext2.clearRect(0, 0, this.latentDrawingCanvas2.width, this.latentDrawingCanvas2.height);
        }
        this.latentURL2 = '';
      }
    },
    toggleCanvasVisibility(): void {
      this.canvasVisible = !this.canvasVisible;
    },
    handleKeyDown(event: { key: string; }) {
      if (event.key === '\\' || event.key === '|') {
        this.downloadCanvasAsImage();
      }
    },
    getKeypointProxy(keypoint: OpenposeKeypoint2D): UnwrapRef<OpenposeKeypoint2D> {
      return this.keypointMap.get(keypoint.id)!;
    },
    updateKeypointProxy(keypoint: OpenposeKeypoint2D) {
      const proxy = this.getKeypointProxy(keypoint);

      proxy.x = keypoint.x;
      proxy.y = keypoint.y;
    },
    addPerson(newPerson: OpenposePerson) {
      this.people.set(newPerson.id, newPerson);
      newPerson.addToCanvas(this.openposeCanvas);
      // Add the reactive keypoints to the keypointMap
      newPerson.allKeypoints().forEach((keypoint) => {
        this.keypointMap.set(keypoint.id, reactive(keypoint));
      });
      this.canvas?.renderAll();
    },
    addDefaultPerson() {
      const newPerson = new OpenposePerson(null, new OpenposeBody(default_body_keypoints));
      this.addPerson(newPerson);
    },
    removePerson(person: OpenposePerson) {
      // If the person is active right now, deactivate it.
      if (person.id.toString() === this.activePersonId) {
        this.updateActivePerson(undefined);
      }

      this.people.delete(person.id);
      person.removeFromCanvas();
      // Remove the reactive keypoints from the keypointMap
      person.allKeypoints().forEach((keypoint) => {
        this.keypointMap.delete(keypoint.id);
      });
      this.canvas?.renderAll();
    },
    addDefaultObject(person: OpenposePerson, part: OpenposeBodyPart) {
      let target: OpenposeObject;
      switch (part) {
        case OpenposeBodyPart.LEFT_HAND:
          target = new OpenposeHand(default_left_hand_keypoints);
          break;
        case OpenposeBodyPart.RIGHT_HAND:
          target = new OpenposeHand(default_right_hand_keypoints);
          break;
        case OpenposeBodyPart.FACE:
          person.face = new OpenposeFace(default_face_keypoints);
          target = person.face;
          break;
      }
      this.addObject(person, part, target);
    },
    addObject(person: OpenposePerson, part: OpenposeBodyPart, target: OpenposeObject) {
      target.addToCanvas(this.openposeCanvas);
      target.keypoints.forEach(keypoint => {
        this.keypointMap.set(keypoint.id, reactive(keypoint));
      });

      switch (part) {
        case OpenposeBodyPart.LEFT_HAND:
          person.attachLeftHand(target as OpenposeHand);
          break;
        case OpenposeBodyPart.RIGHT_HAND:
          person.attachRightHand(target as OpenposeHand);
          break;
        case OpenposeBodyPart.FACE:
          person.attachFace(target as OpenposeFace);
          break;
      }
      this.canvas?.renderAll();
    },
    removeObject(person: OpenposePerson, part: OpenposeBodyPart) {
      let target: OpenposeObject | undefined;
      switch (part) {
        case OpenposeBodyPart.LEFT_HAND:
          target = person.left_hand;
          person.left_hand = undefined;
          break;
        case OpenposeBodyPart.RIGHT_HAND:
          target = person.right_hand;
          person.right_hand = undefined;
          break;
        case OpenposeBodyPart.FACE:
          target = person.face;
          person.face = undefined;
          break;
      }

      if (!target) return;

      target.removeFromCanvas();
      target.keypoints.forEach(keypoint => {
        this.keypointMap.delete(keypoint.id);
      });
      if (this.activeBodyPart === part) {
        this.updateActiveBodyPart(undefined, person);
      }
      this.canvas?.renderAll();
    },
    async waitWindowVisible(): Promise<void> {
      return new Promise((resolve) => {
        const interval = setInterval(() => {
          if (window.innerHeight > 0) {
            clearInterval(interval);
            resolve();
          }
        }, 100);
      });
    },
    resizeHTMLCanvas() {
      const htmlCanvasWidth = Math.round((window.innerWidth * 16 / 24) * 0.95);
      const htmlCanvasHeight = Math.round(window.innerHeight * 0.95);
      this._resizeHTMLCanvas(htmlCanvasWidth, htmlCanvasHeight);
    },
    // HTML canvas is the real canvas object.
    _resizeHTMLCanvas(newWidth: number, newHeight: number) {
      if (!this.canvas)
        return;
      this.canvas.setWidth(newWidth);
      this.canvas.setHeight(newHeight);
      this.canvas.calcOffset();
      this.canvas.requestRenderAll();
    },
    // Openpose canvas is a rect bound on the HTML canvas.
    resizeOpenposeCanvas(newWidth: number, newHeight: number) {
      if (!this.canvas)
        return;
      this.openposeCanvas.set({
        width: newWidth,
        height: newHeight,
      });
      this.canvas.centerObject(this.openposeCanvas);
      this.openposeCanvas.setCoords();
      this.canvas.requestRenderAll();
    },
    onLockedChange(file: LockableUploadFile, locked: boolean) {
      file.locked = locked;
      const img = this.canvasImageMap.get(file.uid);
      if (!img) return;
      if (locked) {
        if (this.canvas?.getActiveObjects().includes(img)) {
          this.canvas.discardActiveObject();
        }
        img.set({
          selectable: false,
          evented: false,
          hasControls: false,
          hasBorders: false,
        });
      } else {
        img.set({
          selectable: true,
          evented: true,
          hasControls: true,
          hasBorders: true,
        });
      }
      this.canvas?.renderAll();
    },
    updateActivePerson(activePersonId: string | undefined) {
      if (this.activePersonId === activePersonId) return;

      if (!activePersonId) {
        // Collapse current panel.
        // If there is a body part panel expanded, collapse that as well.
        const activePerson = this.people.get(parseInt(this.activePersonId!))!;
        this.updateActiveBodyPart(undefined, activePerson);
      }

      this.activePersonId = activePersonId;
    },
    updateActiveBodyPart(activeBodyPart: OpenposeBodyPart | undefined, person: OpenposePerson) {
      if (this.activeBodyPart === activeBodyPart) return;

      if (activeBodyPart === undefined) {
        if (this.activeBodyPart !== undefined) {
          // There can only be one active person. If we collapse the person panel
          // or collapse the body part panel. This function can still receive the 
          // correct person of the targeted object.
          const target = person[this.activeBodyPart]!;
          if (target)
            target.grouped = true;
        }
      } else {
        const target = person[activeBodyPart]!;
        target.grouped = true;
        // Ungroup the object so that user can operate on each individual keypoint.
        target.grouped = false;
      }
      this.activeBodyPart = activeBodyPart;
    },
    resetZoom() {
      if (!this.canvas) return;
      this.canvas.setViewportTransform(IDENTITY_MATRIX);
    },
    /**
     * Adjust canvas zoom level to fit the target group's bounding box.
     * @param group The group object to zoom into.
     * @param zoomed_size: How big should the group take the space of the zoomed canvas.
     */
    zoomToGroup(group: fabric.Group, zoomed_size: number = 1.0) {
      if (!this.canvas) return;

      // Get the bounding rectangle of the group
      const boundingRect = group.getBoundingRect();

      // Calculate the scale factor
      const scaleFactor = Math.min(
        this.canvas.getWidth() / boundingRect.width,
        this.canvas.getHeight() / boundingRect.height
      );

      // Set the zoom level and center the viewport
      const centerX = boundingRect.left + boundingRect.width / 2;
      const centerY = boundingRect.top + boundingRect.height / 2;
      const center = new fabric.Point(centerX, centerY);
      this.canvas.zoomToPoint(center, scaleFactor * zoomed_size);
    },
    handleBeforeUploadImage(file: Blob) {
      const reader = new FileReader();
      reader.onload = (e) => {
        this.loadBackgroundImageFromURL(e.target!.result! as string);
      };
      reader.readAsDataURL(file);
      return false;
    },
    handleUpdateRef(file: Blob, target: number) {
      const reader = new FileReader();
      reader.onload = (e) => {
        if (e.target && e.target.result) {
          const img = new Image();
          img.onload = () => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            ctx.globalAlpha = 0.8;

            if (!ctx) return;

            let width = img.width;
            let height = img.height;
            let paddingX = 0;
            let paddingY = 0;

            if (width > height) {
              paddingY = (width - height) / 2;
            } else if (height > width) {
              paddingX = (height - width) / 2;
            }

            const size = Math.max(width, height);

            canvas.width = size;
            canvas.height = size;

            // Fill the canvas with black
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, size, size);

            // Draw the image on the canvas with padding
            ctx.drawImage(img, paddingX, paddingY, width, height);

            // Resize the canvas to 512x512
            const finalCanvas = document.createElement('canvas');
            const finalCtx = finalCanvas.getContext('2d');

            if (!finalCtx) return;

            finalCanvas.width = 512;
            finalCanvas.height = 512;

            finalCtx.drawImage(canvas, 0, 0, 512, 512);
            if (target == 1)
              this.referenceURL = finalCanvas.toDataURL('image/png');
            else if (target == 2)
              this.referenceURL2 = finalCanvas.toDataURL('image/png');
          };
          img.src = e.target.result as string;
        }
      };
      reader.readAsDataURL(file);
      return false;
    },
    loadBackgroundImageFromURL(url: string) {
      fabric.Image.fromURL(url!, (img) => {
        img.set({
          left: this.openposeCanvas.left,
          top: this.openposeCanvas.top,
          scaleX: 1.0,
          scaleY: 1.0,
          opacity: 1.0,
          hasControls: true,
          hasBorders: true,
          lockScalingX: false,
          lockScalingY: false,
        });

        this.canvas?.add(img);
        // Image should not block skeleton.
        this.canvas?.moveTo(img, 1);
        this.canvas?.renderAll();
        const uploadFile = this.uploadedImageList[this.uploadedImageList.length - 1];
        this.canvasImageMap.set(uploadFile.uid, img);

        uploadFile.locked = true
        this.onLockedChange(uploadFile, true)
      });
    },
    isImage(file: UploadFile) {
      return /\.(jpeg|jpg|gif|png|bmp)$/i.test(file.name);
    },
    handleRemoveImage(image: UploadFile) {
      if (!this.canvasImageMap.has(image.uid)) return;

      this.canvas?.remove(toRaw(this.canvasImageMap.get(image.uid)!));
      this.canvas?.renderAll();
    },
    scaleImage(image: LockableUploadFile, scale: number) {
      image.scale = scale;
      if (!this.canvasImageMap.has(image.uid)) return;

      const img = this.canvasImageMap.get(image.uid)!;
      img.set({
        scaleX: scale,
        scaleY: scale,
      });

      this.canvas?.renderAll();
    },
    readOpenposeJson(file: Blob): Promise<IOpenposeJson> {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
          let poseJson: IOpenposeJson;
          try {
            poseJson = JSON.parse(e.target!.result! as string) as IOpenposeJson;
          } catch (ex: any) {
            reject(ex);
            return;
          }
          resolve(poseJson);
        };
        reader.readAsText(file);
      });
    },
    parseOpenposeJson(poseJson: IOpenposeJson): OpenposePerson[] {
      function preprocessPoints(nums: number[], canvasWidth: number, canvasHeight: number): [number, number, number][] {
        const normalized = _.every(nums, num => Math.abs(num) <= 1.0);
        const xFactor = normalized ? canvasWidth : 1.0;
        const yFactor = normalized ? canvasHeight : 1.0;
        const points = _.chunk(nums, 3) as [number, number, number][];
        return points.map(p => [p[0] * xFactor, p[1] * yFactor, p[2]]);
      }
      const canvasHeight = poseJson.canvas_height;
      const canvasWidth = poseJson.canvas_width;

      return (poseJson.people || []).map((personJson): OpenposePerson | undefined => {
        const body = OpenposeBody.create(preprocessPoints(personJson.pose_keypoints_2d, canvasWidth, canvasHeight));
        if (body === undefined) {
          // If body is malformatted, no need to render face/hand.
          return undefined;
        }
        return new OpenposePerson(null,
          body,
          personJson.hand_left_keypoints_2d ?
            OpenposeHand.create(preprocessPoints(personJson.hand_left_keypoints_2d, canvasWidth, canvasHeight)) : undefined,
          personJson.hand_right_keypoints_2d ?
            OpenposeHand.create(preprocessPoints(personJson.hand_right_keypoints_2d, canvasWidth, canvasHeight)) : undefined,
          personJson.face_keypoints_2d ?
            OpenposeFace.create(preprocessPoints(personJson.face_keypoints_2d, canvasWidth, canvasHeight)) : undefined,
        )
      }).concat(
        (poseJson.animals || []).map((animal): OpenposePerson | undefined => {
          const openposeAnimal = OpenposeAnimal.create(preprocessPoints(animal, canvasWidth, canvasHeight))
          return openposeAnimal ? new OpenposePerson(null, openposeAnimal) : undefined;
        })
      ).filter(person => person !== undefined) as OpenposePerson[];
    },
    /**
     * Adds a body part from the given JSON file.
     * The given JSON is expected to only have one person.
     * @param file JSON file blob
     * @param person 
     * @param part 
     */
    addJsonObject(file: Blob, person: OpenposePerson, part: OpenposeBodyPart) {
      this.readOpenposeJson(file).then(this.parseOpenposeJson).then((people: OpenposePerson[]) => {
        if (people.length === 0) return;
        const firstPerson = people[0];
        switch (part) {
          case OpenposeBodyPart.LEFT_HAND:
            if (firstPerson.left_hand === undefined) {
              this.$notify({ title: 'Error', desc: 'Left hand does not exist in Json' });
              return;
            }
            this.addObject(person, part, firstPerson.left_hand);
            break;
          case OpenposeBodyPart.RIGHT_HAND:
            if (firstPerson.right_hand === undefined) {
              this.$notify({ title: 'Error', desc: 'Right hand does not exist in Json' });
              return;
            }
            this.addObject(person, part, firstPerson.right_hand);
            break;
          case OpenposeBodyPart.FACE:
            if (firstPerson.face === undefined) {
              this.$notify({ title: 'Error', desc: 'Face does not exist in Json' });
              return;
            }
            this.addObject(person, part, firstPerson.face);
            break;
        }
      }).catch((ex) => this.$notify(ex + ''));
      return false;
    },
    /**
     * Adds all people specified in the given JSON file to canvas.
     * @param file JSON file blob
     */
    handleBeforeUploadJson(file: Blob) {
      this.readOpenposeJson(file)
        .then(this.loadPeopleFromJson)
        .catch((ex) => this.$notify(ex + ''));
      return false;
    },
    loadPeopleFromJson(poseJson: IOpenposeJson) {
      const canvasHeight = poseJson.canvas_height;
      const canvasWidth = poseJson.canvas_width;
      this.canvasHeight = _.max([canvasHeight, this.canvasHeight])!;
      this.canvasWidth = _.max([canvasWidth, this.canvasWidth])!;
      this.resizeOpenposeCanvas(this.canvasWidth, this.canvasHeight);
      this.parseOpenposeJson(poseJson).forEach(person => this.addPerson(person));
    },
    /**
     * Clear everything on the canvas.
     */
    clearCanvas() {
      // Remove all people.
      [...this.people.values()].forEach(person => {
        this.removePerson(person);
      });
      // Remove all background images.
      this.uploadedImageList.forEach(image => {
        this.handleRemoveImage(image);
      });
      this.uploadedImageList.splice(0); // Clear `uploadedImageList`.
      this.resetZoom();
    },
    async loadCanvasFromFrameMessage(message: IncomingFrameMessage) {
      this.modalId = message.modalId;

      this.clearCanvas();
      const openposeJson =
        message.poseURL ?
          parseDataURLtoJSON(message.poseURL) as IOpenposeJson :
          Array.isArray(message.poses!) ? message.poses![0] : message.poses!;

      this.canvasHeight = openposeJson.canvas_height;
      this.canvasWidth = openposeJson.canvas_width;
      this.loadPeopleFromJson(openposeJson);

      // (Optional) Loads background image.
      if (message.imageURL) {
        const imageFile = {
          locked: false,
          scale: 1.0,
          name: 'controlnet input',
          uid: await calculateHash(message.imageURL),
          upload_url: message.imageURL
        } as LockableUploadFile;
        this.uploadedImageList.push(imageFile);
        this.loadBackgroundImageFromURL(message.imageURL);
        const [imgWidth, imgHeight] = await getImageDimensionsFromDataURL(message.imageURL);
        this.scaleImage(imageFile, Math.min(this.canvasHeight / imgHeight, this.canvasWidth / imgWidth));
        imageFile.locked = true;
        this.onLockedChange(imageFile, true);
      }
    },
    getCanvasAsOpenposeJson(): IOpenposeJson {
      return {
        people: [...this.people.values()]
          .filter(person => !person.allKeypointsInvisible() && !person.isAnimal)
          .map(person => person.toJson()),
        animals: [...this.people.values()]
          .filter(person => !person.allKeypointsInvisible() && person.isAnimal)
          .map(person => person.toJson()),
        canvas_width: this.canvasWidth,
        canvas_height: this.canvasHeight,
      } as IOpenposeJson;
    },
    sendCanvasAsFrameMessage() {
      if (this.modalId === undefined) return;
      const poses = this.getCanvasAsOpenposeJson();
      window.parent.postMessage({
        modalId: this.modalId,
        poseURL: serializeJSONtoDataURL(poses),
        poses: poses,
      } as OutgoingFrameMessage, '*');
    },
    downloadCanvasAsJson() {
      const link = document.createElement('a');
      link.href = serializeJSONtoDataURL(this.getCanvasAsOpenposeJson());
      link.download = 'pose.json';
      link.click();
    },
    async downloadCanvasAsImage() {
      // Clear uploaded images list
      for (const image of this.uploadedImageList) {
        this.temp_img = image;
        this.handleRemoveImage(image);
      }
      this.uploadedImageList = [];

      // Ensure canvas is available
      if (!this.canvas) return;
      this.resetZoom();

      // Get the data URL of the canvas as a PNG image
      const dataUrl = this.canvas.toDataURL({ format: 'image/png' });

      // Create a new canvas for cropping
      const newCanvas = new fabric.StaticCanvas(null, {
        width: this.canvasWidth,
        height: this.canvasHeight,
      });

      // Re-upload previously removed image
      if (this.temp_img != null) {
        const imageFile = {
          locked: false,
          scale: 1.0,
          name: 'controlnet input',
          uid: await calculateHash(this.temp_img.upload_url!.toString()),
          upload_url: this.temp_img.upload_url,
        } as LockableUploadFile;
        this.uploadedImageList.push(imageFile);
        this.loadBackgroundImageFromURL(this.temp_img.upload_url!.toString());
      }

      // Crop and process the image if openposeCanvas is available
      if (this.openposeCanvas) {
        fabric.Image.fromURL(dataUrl, img => {
          img.set({
            left: -this.openposeCanvas.left!,
            top: -this.openposeCanvas.top!,
          });

          newCanvas.add(img);
          newCanvas.renderAll();

          // Get the cropped image URL
          const croppedImageUrl = newCanvas.toDataURL({ format: 'image/png' });
          this.conditionURL = croppedImageUrl

          console.log([
            this.num_inference_steps,
            this.seed,
            this.pos,
            this.neg,
            this.canvasWidth,
            this.canvasHeight,
            this.img_p_scale,
            this.ctrl_scale,
            this.cfg,
            croppedImageUrl,
            this.referenceURL,
            this.maskURL,
            this.latentURL,
            this.referenceURL2,
            this.maskURL2,
            this.latentURL2
          ])

          fetch("http://localhost:7860/api/generate", {
            method: "POST",
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              data: [
                this.num_inference_steps,
                this.seed,
                this.pos,
                this.neg,
                this.canvasWidth,
                this.canvasHeight,
                this.img_p_scale,
                this.ctrl_scale,
                this.cfg,
                croppedImageUrl,
                this.referenceURL,
                this.maskURL,
                this.latentURL,
                this.referenceURL2,
                this.maskURL2,
                this.latentURL2
              ]
            })
          }).then(res => res.json())
            .then(async data => {
              this.uploadedImageList.forEach(image => {
                this.handleRemoveImage(image);
              });
              this.uploadedImageList = []

              let newImageURL = data.data[0].url.toString();

              console.log(newImageURL)
              this.imageURL = newImageURL;

              var imageFile = {
                locked: false,
                scale: 1.0,
                name: 'controlnet input',
                uid: await calculateHash(newImageURL),
                upload_url: newImageURL
              } as LockableUploadFile;
              this.uploadedImageList.push(imageFile);
              this.loadBackgroundImageFromURL(newImageURL);
            })
            .catch(error => {
              console.error('Error:', error);  // Handle any errors
            });
        });
      }
    }
  },

  components: {
    PlusSquareOutlined,
    UploadOutlined,
    OpenposeObjectPanel,
  }
});
</script>

<template>
  <a-row>
    <a-col :span="8" id="control-panel">

      <a-divider orientation="left" orientation-margin="0px">
        Generate by key "\"
      </a-divider>
      <a-space>
        <a-button @click="downloadCanvasAsImage" type="primary" size="large">
          Generate
        </a-button>
      </a-space>

      <a-divider orientation="left" orientation-margin="0px">
        Reference Image
      </a-divider>
      <a-upload list-type="picture" accept="image/*" :beforeUpload="file => handleUpdateRef(file, 1)"
        :show-upload-list="false">
        <a-button>
          <upload-outlined></upload-outlined>
          Upload Reference
        </a-button>
      </a-upload>

      <div>
        <div
          :style="{ 'height': '512px', 'width': '512px', 'background-color': 'black', position: 'absolute', top: 0, left: 0 }">
        </div>
        <a-image :style="{ position: 'absolute', top: 0, left: 0 }" :src="referenceURL.toString()" alt="avatar"
          :width="512" :height="512" :preview="false" />
        <div :style="{ position: 'absolute', top: 0, left: 0, opacity: 0.8 }">
          <!-- <div
            :style="{ 'height': '512px', 'width': '512px', 'background-color': 'black', position: 'absolute', top: 0, left: 0 }">
          </div> -->
          <canvas ref="maskDrawingCanvas" :width="512" :height="512"
            @mousedown="startDrawingOnMaskCanvas($event, 'mask')" @mousemove="drawOnMaskCanvas($event, 'mask')"
            @mouseup="stopDrawingMaskOnCanvas('mask')" @mouseleave="stopDrawingMaskOnCanvas('mask')"></canvas>
        </div>
      </div>
      <a-space>
        <!-- <a-button @click="toggleCanvasVisibility">{{ canvasVisible ? 'Hide' : 'Show' }} Mask</a-button> -->
        <a-button @click="clearDrawingCanvas('mask')" danger>Clear Reference Mask</a-button>
      </a-space>

      <a-divider />
      <a-button @click="latentVisible = !latentVisible">
        {{ latentVisible ? 'Hide' : 'Show' }} Latent Mask
      </a-button>
      <div v-show="latentVisible">
        <div>
          <div
            :style="{ 'height': '512px', 'width': '512px', 'background-color': 'black', position: 'absolute', top: 0, left: 0 }">
          </div>
          <a-image :style="{ position: 'absolute', top: 0, left: 0 }" :src="conditionURL.toString()" alt="avatar"
            :width="512" :height="512" :preview="false" />
          <div :style="{ position: 'absolute', top: 0, left: 0, opacity: 0.8 }">
            <canvas ref="latentDrawingCanvas" :width="512" :height="512"
              @mousedown="startDrawingOnMaskCanvas($event, 'latent')" @mousemove="drawOnMaskCanvas($event, 'latent')"
              @mouseup="stopDrawingMaskOnCanvas('latent')" @mouseleave="stopDrawingMaskOnCanvas('latent')"></canvas>
          </div>
        </div>

        <a-space>
          <a-button @click="clearDrawingCanvas('latent')" danger>Clear Latent Mask</a-button>
        </a-space>
      </div>
      <a-divider />
      <a-divider orientation="left" orientation-margin="0px"> 2nd Reference Image </a-divider>
      <a-button @click="reference2Visible = !reference2Visible">
        {{ reference2Visible ? 'Hide' : 'Show' }} 2nd Reference Image
      </a-button>

      <div v-show="reference2Visible">
        <a-divider />
        <a-upload list-type="picture" accept="image/*" :beforeUpload="file => handleUpdateRef(file, 2)"
          :show-upload-list="false">
          <a-button>
            <upload-outlined></upload-outlined>
            Upload 2nd Reference
          </a-button>
        </a-upload>

        <div>
          <div
            :style="{ 'height': '512px', 'width': '512px', 'background-color': 'black', position: 'absolute', top: 0, left: 0 }">
          </div>
          <a-image :style="{ position: 'absolute', top: 0, left: 0 }" :src="referenceURL2.toString()" alt="avatar"
            :width="512" :height="512" :preview="false" />
          <div :style="{ position: 'absolute', top: 0, left: 0, opacity: 0.8 }">
            <canvas ref="maskDrawingCanvas2" :width="512" :height="512"
              @mousedown="startDrawingOnMaskCanvas($event, 'mask2')" @mousemove="drawOnMaskCanvas($event, 'mask2')"
              @mouseup="stopDrawingMaskOnCanvas('mask2')" @mouseleave="stopDrawingMaskOnCanvas('mask2')"></canvas>
          </div>
        </div>
        <a-space>
          <a-button @click="clearDrawingCanvas('mask2')" danger>Clear Reference Mask</a-button>
        </a-space>
        <a-divider />

        <div>
          <div
            :style="{ 'height': '512px', 'width': '512px', 'background-color': 'black', position: 'absolute', top: 0, left: 0 }">
          </div>
          <a-image :style="{ position: 'absolute', top: 0, left: 0 }" :src="conditionURL.toString()" alt="avatar"
            :width="512" :height="512" :preview="false" />
          <div :style="{ position: 'absolute', top: 0, left: 0, opacity: 0.8 }">
            <canvas ref="latentDrawingCanvas2" :width="512" :height="512"
              @mousedown="startDrawingOnMaskCanvas($event, 'latent2')" @mousemove="drawOnMaskCanvas($event, 'latent2')"
              @mouseup="stopDrawingMaskOnCanvas('latent2')" @mouseleave="stopDrawingMaskOnCanvas('latent2')"></canvas>
          </div>
        </div>

        <a-space>
          <a-button @click="clearDrawingCanvas('latent2')" danger>Clear Latent Mask</a-button>
        </a-space>
        
        <a-divider />
      </div>

      <a-divider orientation="left" orientation-margin="0px">
        {{ $t('ui.canvas') }}
      </a-divider>
      <div>
        <a-space>
          <a-input-number type="inputNumber" addon-before="Width" addon-after="px" v-model:value="canvasWidth" :min="64"
            :max="4096" />
          <a-input-number type="inputNumber" addon-before="Height" addon-after="px" v-model:value="canvasHeight"
            :min="64" :max="4096" />
        </a-space>
        <a-space>
          <a-button @click="resizeOpenposeCanvas(canvasWidth, canvasHeight)">{{ $t('ui.resizeCanvas') }}</a-button>
          <a-button @click="resetZoom()">{{ $t('ui.resetZoom') }}</a-button>
        </a-space>
      </div>

      <a-divider orientation="left" orientation-margin="0px">
        {{ $t('ui.poseControl') }}
      </a-divider>
      <a-space>
        <a-button @click="addDefaultPerson">
          <plus-square-outlined />
          {{ $t('ui.addPerson') }}
        </a-button>
      </a-space>
      <a-collapse accordion :activeKey="activePersonId" @update:activeKey="updateActivePerson">
        <OpenposeObjectPanel v-for="person in people.values()" :object="person.body" :display_name="person.name"
          @removeObject="removePerson(person)" :key="person.id">
          <template #extra-control>
            <!-- TODO: make this repetitive code a component. -->
            <div v-if="person.left_hand === undefined && !person.isAnimal">
              <a-button @click="addDefaultObject(person, OpenposeBodyPart.LEFT_HAND)">{{ $t('ui.addLeftHand')
                }}</a-button>
              <a-upload accept="application/json"
                :beforeUpload="(file: Blob) => addJsonObject(file, person, OpenposeBodyPart.LEFT_HAND)"
                :showUploadList="false">
                <a-button>
                  <upload-outlined></upload-outlined>
                </a-button>
              </a-upload>
            </div>
            <div v-if="person.right_hand === undefined && !person.isAnimal">
              <a-button @click="addDefaultObject(person, OpenposeBodyPart.RIGHT_HAND)">{{ $t('ui.addRightHand')
                }}</a-button>
              <a-upload accept="application/json"
                :beforeUpload="(file: Blob) => addJsonObject(file, person, OpenposeBodyPart.RIGHT_HAND)"
                :showUploadList="false">
                <a-button>
                  <upload-outlined></upload-outlined>
                </a-button>
              </a-upload>
            </div>
            <div v-if="person.face === undefined && !person.isAnimal">
              <a-upload accept="application/json"
                :beforeUpload="(file: Blob) => addJsonObject(file, person, OpenposeBodyPart.FACE)"
                :showUploadList="false">
                <a-button>
                  {{ $t('ui.addFace') }}
                  <upload-outlined></upload-outlined>
                </a-button>
              </a-upload>
            </div>
            <a-collapse accordion :activeKey="activeBodyPart" @update:activeKey="updateActiveBodyPart($event, person)">
              <OpenposeObjectPanel v-if="person.left_hand !== undefined" :object="person.left_hand"
                :display_name="'Left Hand'" @removeObject="removeObject(person, OpenposeBodyPart.LEFT_HAND)"
                :key="OpenposeBodyPart.LEFT_HAND" />
              <OpenposeObjectPanel v-if="person.right_hand !== undefined" :object="person.right_hand"
                :display_name="'Right Hand'" @removeObject="removeObject(person, OpenposeBodyPart.RIGHT_HAND)"
                :key="OpenposeBodyPart.RIGHT_HAND" />
              <OpenposeObjectPanel v-if="person.face !== undefined" :object="person.face" :display_name="'Face'"
                @removeObject="removeObject(person, OpenposeBodyPart.FACE)" :key="OpenposeBodyPart.FACE" />
            </a-collapse>
          </template>
        </OpenposeObjectPanel>
      </a-collapse>


      <a-divider orientation="left" orientation-margin="0px">
        Advance Settings
      </a-divider>
      <a-space direction="vertical" style="width:100%">
        <a-form-item label="Positive Prompt">
          <a-input v-model:value="pos" />
        </a-form-item>
        <a-form-item label="Negative Prompt">
          <a-input v-model:value="neg" />
        </a-form-item>
        <a-form-item label="Steps">
          <a-input v-model:value="num_inference_steps" />
        </a-form-item>
        <a-form-item label="Seed">
          <a-input v-model:value="seed" />
        </a-form-item>
        <a-form-item label="Reference Scale">
          <a-input v-model:value="img_p_scale" />
        </a-form-item>
        <a-form-item label="Control Scale">
          <a-input v-model:value="ctrl_scale" />
        </a-form-item>
        <a-form-item label="CFG">
          <a-input v-model:value="cfg" />
        </a-form-item>
      </a-space>
    </a-col>

    <a-col :span="16" id="canvas-panel">
      <canvas ref="editorCanvas"></canvas>
    </a-col>
  </a-row>
</template>

<style>
.hidden {
  opacity: 50%;
  text-decoration: line-through;
}
</style>
