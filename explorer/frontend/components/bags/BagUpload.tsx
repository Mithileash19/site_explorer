"use client";

import { useRef, useState } from "react";
import { uploadBag } from "@/lib/api";
import { UploadCloud, File as FileIcon, CheckCircle, XCircle, Loader2 } from "lucide-react";

interface Props {
  onUploaded: (bagPath: string) => void;
}

type Status = "idle" | "uploading" | "done" | "error";

export default function BagUpload({ onUploaded }: Props) {
  const inputRef        = useRef<HTMLInputElement>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [msg,    setMsg]    = useState("");
  const [drag,   setDrag]   = useState(false);

  async function handleFile(file: File) {
    if (!file.name.match(/\.(bag|db3)$/i)) {
      setMsg("Only .bag and .db3 files are accepted.");
      setStatus("error");
      return;
    }
    setStatus("uploading");
    setMsg(`Uploading ${file.name} …`);
    try {
      const res = await uploadBag(file);
      setMsg(`Uploaded  ${file.name}  (${res.size_mb.toFixed(1)} MB)`);
      setStatus("done");
      onUploaded(res.bag_path);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Upload failed";
      setMsg(msg);
      setStatus("error");
    }
  }

  function onInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDrag(false);
    const f = e.dataTransfer.files?.[0];
    if (f) handleFile(f);
  }

  return (
    <div>
      <div
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`
          flex flex-col items-center justify-center gap-3 rounded-xl
          border-2 border-dashed p-10 cursor-pointer transition-colors
          ${drag ? "border-blue-500 bg-blue-500/5" : "border-slate-700 hover:border-slate-600 bg-slate-800/30"}
        `}
      >
        <UploadCloud
          size={36}
          className={drag ? "text-blue-400" : "text-slate-500"}
        />
        <div className="text-center">
          <p className="text-sm font-medium text-slate-300">
            {drag ? "Drop to upload" : "Drag & drop your ROS bag here"}
          </p>
          <p className="text-xs text-slate-500 mt-1">or click to browse · .bag .db3 · max 400 MB</p>
        </div>
        <input
          ref={inputRef}
          type="file"
          accept=".bag,.db3"
          className="hidden"
          onChange={onInputChange}
        />
      </div>

      {status !== "idle" && (
        <div className={`
          flex items-center gap-2 mt-3 px-3 py-2 rounded-lg text-sm
          ${status === "uploading" ? "text-blue-300 bg-blue-900/20"
           : status === "done"     ? "text-green-300 bg-green-900/20"
           :                          "text-red-300 bg-red-900/20"}
        `}>
          {status === "uploading" && <Loader2 size={14} className="animate-spin shrink-0" />}
          {status === "done"      && <CheckCircle size={14} className="shrink-0" />}
          {status === "error"     && <XCircle     size={14} className="shrink-0" />}
          {status === "done" && <FileIcon size={14} className="shrink-0" />}
          <span className="truncate">{msg}</span>
        </div>
      )}
    </div>
  );
}
