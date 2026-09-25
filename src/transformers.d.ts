/**
 * Minimal ambient declaration for the prebuilt transformers.js browser bundle.
 * Only the surface this project uses is declared, so misuse is still caught.
 */
declare module '*/transformers.web.js' {
  export const env: any;
  export const AutoModel: { from_pretrained(id: string, opts?: any): Promise<any> };
  export const AutoProcessor: { from_pretrained(id: string, opts?: any): Promise<any> };
  export class RawImage {
    constructor(data: Uint8ClampedArray, w: number, h: number, ch: number);
  }
}
