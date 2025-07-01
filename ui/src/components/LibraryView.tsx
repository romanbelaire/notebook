import { useState, useEffect, useRef } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  listPapers,
  listCollections,
  listInsights,
  createCollection as apiCreateCollection,
  addPapersToCollection as apiAddToCollection,
  deleteInsight,
  deleteNote,
  removeFromCollection as apiRemoveFromCollection,
  renameCollection as apiRenameCollection,
  removeCollection as apiRemoveCollection,
  clearDatabase,
} from "../api";
import type { Paper, Collection, Insight } from "../api";
import PlusIcon from "../assets/plus.svg?react";
import TrashIcon from "../assets/trash.svg?react";
import CloseIcon from "../assets/close.svg?react";
import MagnifyIcon from "../assets/magnify.svg?react";
import BookIcon from "../assets/book.svg?react";
import PencilIcon from "../assets/pencil.svg?react";
import { useUIStore } from "../store/ui";
import { useInsightsStore } from "../store/insights";

export default function LibraryView() {
  const queryClient = useQueryClient();

  // Fetch papers, collections & insights
  const { data: papers } = useQuery<Paper[]>({ queryKey: ["papers"], queryFn: listPapers });
  const { data: collections } = useQuery<Collection[]>({ queryKey: ["collections"], queryFn: listCollections });
  const { data: insights } = useQuery<Insight[]>({ queryKey: ["insights"], queryFn: listInsights });

  // Local UI state
  const [selectedPapers, setSelectedPapers] = useState<Set<number>>(new Set());
  const [selectedInsights, setSelectedInsights] = useState<Set<string>>(new Set());
  const [newCollName, setNewCollName] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCollId, setSelectedCollId] = useState<number | null>(null);
  const newCollInputRef = useRef<HTMLInputElement | null>(null);
  const [showCollDropdown, setShowCollDropdown] = useState(false);
  const [editingCollName, setEditingCollName] = useState(false);
  const [newCollectionName, setNewCollectionName] = useState("");
  const [deleteConfirmColl, setDeleteConfirmColl] = useState(false);
  const dropdownRef = useRef<HTMLDivElement | null>(null);

  // Focus management: if ChatWindow requested focusing this input, do so
  const focusNewCollection = useUIStore((s) => s.focusNewCollection);
  const clearFocusNewCollection = useUIStore((s) => s.clearFocusNewCollection);

  // Modal setter for insights
  const setModalInsight = useInsightsStore((s) => s.setModalInsight);

  useEffect(() => {
    if (focusNewCollection) {
      // Give the DOM a tick to ensure the input is rendered
      setTimeout(() => {
        newCollInputRef.current?.focus();
      }, 0);
      clearFocusNewCollection();
    }
  }, [focusNewCollection]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setShowCollDropdown(false);
      }
    };
    if (showCollDropdown) {
      window.addEventListener("mousedown", handleClickOutside);
    }
    return () => {
      window.removeEventListener("mousedown", handleClickOutside);
    };
  }, [showCollDropdown]);

  // When viewing a collection, select all papers in that collection
  useEffect(() => {
    if (selectedCollId && collections) {
      const collection = collections.find(c => c.id === selectedCollId);
      if (collection && collection.papers) {
        setSelectedPapers(new Set(collection.papers.map(p => p.id)));
      }
    }
  }, [selectedCollId, collections]);

  // Split papers into real PDFs and note files (md/txt/docx)
  const paperFiles = (papers ?? []).filter((p) => !/\.(md|markdown|txt|docx)$/i.test(p.filename));
  const noteFiles = (papers ?? []).filter((p) => /\.(md|markdown|txt|docx)$/i.test(p.filename));

  // Helper for case-insensitive substring match
  const matchesSearch = (text: string | undefined) =>
    searchQuery.trim() === "" ? true : (text || "").toLowerCase().includes(searchQuery.trim().toLowerCase());

  // Get current collection's papers if one is selected
  const currentCollection = selectedCollId && collections ? collections.find(c => c.id === selectedCollId) : null;
  const collectionPaperIds = currentCollection && currentCollection.papers
    ? new Set(currentCollection.papers.map((p) => p.id))
    : null;

  // Apply collection and search filtering
  const filteredPaperFiles = paperFiles
    .filter(p => !collectionPaperIds || collectionPaperIds.has(p.id))
    .filter(p => matchesSearch(p.title) || matchesSearch(p.filename));
  
  const filteredNoteFiles = noteFiles
    .filter(p => !collectionPaperIds || collectionPaperIds.has(p.id))
    .filter(p => matchesSearch(p.title) || matchesSearch(p.filename));
    
  const filteredInsights = (insights ?? []).filter((ins) => matchesSearch(ins.title) || matchesSearch(ins.text));

  // Collapsed state for each sub-section
  const [collapsed, setCollapsed] = useState<{ papers: boolean; notes: boolean; insights: boolean }>({
    papers: false,
    notes: false,
    insights: false,
  });

  const toggleCollapse = (key: keyof typeof collapsed) =>
    setCollapsed((prev) => ({ ...prev, [key]: !prev[key] }));

  const toggleSelectPaper = (id: number) => {
    setSelectedPapers((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleSelectInsight = (id: string) => {
    setSelectedInsights((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  // Mutations
  const createCollectionMut = useMutation<Collection, Error, string>({
    mutationFn: apiCreateCollection,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["collections"] });
      setNewCollName("");
    },
  });

  const addMut = useMutation<void, Error, { collId: number; paperIds: number[] }>({
    mutationFn: ({ collId, paperIds }) => apiAddToCollection(collId, paperIds),
    onSuccess: () => {
      setSelectedPapers(new Set());
    },
  });

  // Delete selected notes & insights
  const deleteMut = useMutation<void, Error, void>({
    mutationFn: async () => {
      const noteIds = Array.from(selectedPapers).filter((id) => noteFiles.some((p) => p.id === id));
      const insIds = Array.from(selectedInsights);
      await Promise.all([
        ...noteIds.map((nid) => deleteNote(nid)),
        ...insIds.map((iid) => deleteInsight(iid)),
      ]);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["papers"] });
      queryClient.invalidateQueries({ queryKey: ["insights"] });
      setSelectedPapers(new Set());
      setSelectedInsights(new Set());
    },
  });

  // Collection mutations
  const removeFromCollMutation = useMutation<void, Error, { collId: number; paperIds: number[] }>({
    mutationFn: ({ collId, paperIds }) => apiRemoveFromCollection(collId, paperIds),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["collections"] });
    },
  });

  const renameCollectionMutation = useMutation<void, Error, { collId: number; name: string }>({
    mutationFn: ({ collId, name }) => apiRenameCollection(collId, name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["collections"] });
      setEditingCollName(false);
    },
  });

  const deleteCollectionMutation = useMutation<void, Error, number>({
    mutationFn: (collId) => apiRemoveCollection(collId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["collections"] });
      setSelectedCollId(null);
      setDeleteConfirmColl(false);
    },
  });

  // Toolbar dropdown state
  const [showAddMenu, setShowAddMenu] = useState(false);

  // Compute selection boolean early for hooks below
  const anySelected = selectedPapers.size > 0 || selectedInsights.size > 0;

  // Delete confirmation state
  const [deleteConfirm, setDeleteConfirm] = useState(false);

  // Reset confirmation when selection cleared
  useEffect(() => {
    if (!anySelected) {
      setDeleteConfirm(false);
    }
  }, [anySelected]);

  // ------------------------------------------------------------------
  // Debug: Clear database (dangerous!)
  // ------------------------------------------------------------------

  const [clearDbConfirm, setClearDbConfirm] = useState(false);

  const clearDbMutation = useMutation<void, Error, void>({
    mutationFn: clearDatabase,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["papers"] });
      queryClient.invalidateQueries({ queryKey: ["collections"] });
      queryClient.invalidateQueries({ queryKey: ["insights"] });
      setSelectedPapers(new Set());
      setSelectedInsights(new Set());
      setSelectedCollId(null);
      setClearDbConfirm(false);
    },
  });

  return (
    <div className="p-8 overflow-y-auto">
      {/* Background card */}
      <div className="bg-secondaryBg rounded-lg p-8 space-y-6">
        {/* Documents list – scrollable as a whole */}
        <div className="flex flex-col gap-4 text-lg font-semibold">Documents</div>
        <div
          id="documents-list"
          className="space-y-6 overflow-auto max-h-[440px] pr-2 shadow-inner-strong border border-none rounded"
        >
          {/* Toolbar (sticky inside, pill style) */}
          
          <div className="sticky text-defaultText top-0 left-0 right-0 flex items-center gap-4 px-4 py-1 bg-chat-assistant-bg/20 backdrop-blur-md z-10 rounded-full shadow mx-[6ch] mt-1">
            {/* Search bar */}
            <div className="relative flex items-center text-defaultText">
              <MagnifyIcon className="w-6 h-6 absolute left-1 pointer-events-none" />
              <input
                type="text"
                placeholder="Search..."
                className="pl-7 pr-2 py-1 rounded-full bg-buttonBg text-defaultText placeholder:text-defaultText/60 focus:outline-none"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>

            {/* Collection selector */}
            <div className="relative" ref={dropdownRef}>
              <button
                className="bg-buttonBg text-defaultText px-3 py-1 rounded flex items-center justify-center"
                onClick={() => setShowCollDropdown(!showCollDropdown)}
              >
                <BookIcon className="w-4 h-4" />
              </button>
              {showCollDropdown && collections && (
                <ul className="absolute mt-1 left-0 bg-primaryBg border border-trim/40 rounded shadow-lg z-30 min-w-[10rem]">
                  <li key="all-docs">
                    <button
                      className="w-full text-left px-3 py-1 hover:bg-trim/20"
                      onClick={() => {
                        setSelectedCollId(null);
                        setShowCollDropdown(false);
                        setSelectedPapers(new Set());
                      }}
                    >
                      All Documents
                    </button>
                  </li>
                  {collections.map((c) => (
                    <li key={c.id}>
                      <button
                        className="w-full text-left px-3 py-1 hover:bg-trim/20"
                        onClick={() => {
                          setSelectedCollId(c.id);
                          setShowCollDropdown(false);
                          setSelectedPapers(new Set());
                        }}
                      >
                        {c.name}
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <div className="relative">
                <button
                  className="bg-buttonBg text-defaultText px-3 py-1 rounded disabled:opacity-50 "
                  disabled={selectedPapers.size === 0 || addMut.isPending || !collections?.length}
                  onClick={() => setShowAddMenu((prev) => !prev)}
                >
                  <PlusIcon className="w-4 h-4" />
                </button>
                {showAddMenu && collections && (
                  <ul className="absolute mt-1 right-0 bg-primaryBg border border-trim/40 rounded shadow-lg z-30 min-w-[10rem]">
                    {collections.map((c) => (
                      <li key={c.id}>
                        <button
                          className="w-full text-left px-3 py-1 hover:bg-trim/20"
                          onClick={() => {
                            setShowAddMenu(false);
                            if (selectedPapers.size > 0) {
                              addMut.mutate({ collId: c.id, paperIds: Array.from(selectedPapers) });
                            }
                          }}
                        >
                          {c.name}
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
              {deleteConfirm ? (
                <>
                  <button
                    className="flex px-2 py-1 h-7 items-center justify-center text-xs rounded bg-[#db363c]/50 hover:bg-[#db363c] focus:outline-none"
                    disabled={deleteMut.isPending}
                    onClick={() => {
                      deleteMut.mutate();
                      setDeleteConfirm(false);
                    }}
                    title="Confirm delete"
                  >
                    Delete
                  </button>
                  <button
                    className="w-7 h-7 p-0 flex items-center justify-center bg-transparent border-0 hover:bg-white/10 focus:outline-none rounded"
                    onClick={() => setDeleteConfirm(false)}
                    title="Cancel"
                  >
                    <CloseIcon className="w-4 h-4 pointer-events-none" />
                  </button>
                </>
              ) : (
                <button
                  className="bg-buttonBg text-defaultText px-3 py-1 rounded disabled:opacity-50"
                  disabled={!anySelected || deleteMut.isPending}
                  onClick={() => setDeleteConfirm(true)}
                  title="Delete selected"
                >
                  <TrashIcon className="w-4 h-4" />
                </button>
              )}
          </div>

          {/* Collection header is now rendered inside inner list below */}

          <div className="pl-4 pr-4 space-y-6">
            {/* Collection title & actions */}
            {selectedCollId && collections && (
              <div className="flex items-center gap-4 py-1">
                {editingCollName ? (
                  <>
                    <input
                      type="text"
                      className="flex-1 px-2 py-1 rounded bg-buttonBg text-defaultText"
                      value={newCollectionName}
                      onChange={(e) => setNewCollectionName(e.target.value)}
                      autoFocus
                    />
                    <button
                      className="px-3 py-0 rounded bg-buttonBg text-defaultText"
                      disabled={renameCollectionMutation.isPending || !newCollectionName.trim()}
                      onClick={() => {
                        if (!newCollectionName.trim() || !selectedCollId) {
                          setEditingCollName(false);
                          return;
                        }
                        renameCollectionMutation.mutate(
                          { collId: selectedCollId, name: newCollectionName.trim() },
                          {
                            onSuccess: () => {
                              setEditingCollName(false);
                            },
                          }
                        );
                      }}
                    >
                      Save
                    </button>
                    <button
                      className="w-7 h-7 p-0 flex items-center justify-center rounded hover:bg-white/10"
                      onClick={() => setEditingCollName(false)}
                    >
                      <CloseIcon className="w-4 h-4" />
                    </button>
                  </>
                ) : (
                  <>
                    <h2 className="flex-1 font-medium text-lg">
                      {collections.find((c) => c.id === selectedCollId)?.name}
                    </h2>
                    <button
                      className="w-7 h-7 p-0 flex items-center justify-center rounded hover:bg-white/10 text-defaultText"
                      onClick={() => {
                        const coll = collections.find((c) => c.id === selectedCollId);
                        if (coll) {
                          setNewCollectionName(coll.name);
                          setEditingCollName(true);
                        }
                      }}
                    >
                      <PencilIcon className="w-4 h-4" />
                    </button>
                    {deleteConfirmColl ? (
                      <>
                        <button
                          className="px-2 py-1 text-xs rounded bg-[#db363c]/50 hover:bg-[#db363c]"
                          onClick={() => {
                            if (selectedCollId) {
                              deleteCollectionMutation.mutate(selectedCollId);
                            }
                          }}
                        >
                          Delete
                        </button>
                        <button
                          className="w-7 h-7 p-0 flex items-center justify-center rounded hover:bg-white/10"
                          onClick={() => setDeleteConfirmColl(false)}
                        >
                          <CloseIcon className="w-4 h-4" />
                        </button>
                      </>
                    ) : (
                      <button
                        className="w-7 h-7 p-0 flex items-center justify-center rounded hover:bg-white/10 text-defaultText"
                        onClick={() => setDeleteConfirmColl(true)}
                      >
                        <TrashIcon className="w-4 h-4" />
                      </button>
                    )}
                  </>
                )}
              </div>
            )}

            {/* Papers Section */}
            <section>
              <button
                className="font-medium mb-2 flex items-center gap-1 bg-transparent border-0 p-0 hover:bg-transparent focus:outline-none"
                onClick={() => toggleCollapse("papers")}
              >
                <span>{collapsed.papers ? "▶" : "▼"}</span> Papers 
                {currentCollection ? (
                  <span className="text-gray-500 text-sm font-normal ml-1">
                    ({filteredPaperFiles.length} in collection)
                  </span>
                ) : (
                  <span>({filteredPaperFiles.length})</span>
                )}
              </button>
              {!collapsed.papers && (
                filteredPaperFiles.length > 0 ? (
                  <ul className="space-y-1">
                    {filteredPaperFiles.map((p) => (
                      <li key={p.id} className="flex items-center gap-2">
                        <input 
                          type="checkbox" 
                          checked={selectedPapers.has(p.id)} 
                          onChange={() => {
                            if (selectedCollId) {
                              // If viewing a collection, unchecking removes from collection
                              if (selectedPapers.has(p.id)) {
                                removeFromCollMutation.mutate({ 
                                  collId: selectedCollId, 
                                  paperIds: [p.id] 
                                });
                              }
                            }
                            toggleSelectPaper(p.id);
                          }} 
                        />
                        <span className="truncate" title={p.filename}>
                          {p.title ? `${p.title} (${p.filename})` : p.filename}
                        </span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-gray-500">No papers ingested yet.</p>
                )
              )}
            </section>

            {/* Notes Section */}
            <section>
              <button
                className="font-medium mb-2 flex items-center gap-1 bg-transparent border-0 p-0 hover:bg-transparent focus:outline-none"
                onClick={() => toggleCollapse("notes")}
              >
                <span>{collapsed.notes ? "▶" : "▼"}</span> Notes 
                {currentCollection ? (
                  <span className="text-gray-500 text-sm font-normal ml-1">
                    ({filteredNoteFiles.length} in collection)
                  </span>
                ) : (
                  <span>({filteredNoteFiles.length})</span>
                )}
              </button>
              {!collapsed.notes && (
                filteredNoteFiles.length > 0 ? (
                  <ul className="space-y-1">
                    {filteredNoteFiles.map((p) => (
                      <li key={p.id} className="flex items-center gap-2">
                        <input 
                          type="checkbox" 
                          checked={selectedPapers.has(p.id)} 
                          onChange={() => {
                            if (selectedCollId) {
                              // If viewing a collection, unchecking removes from collection
                              if (selectedPapers.has(p.id)) {
                                removeFromCollMutation.mutate({ 
                                  collId: selectedCollId, 
                                  paperIds: [p.id] 
                                });
                              }
                            }
                            toggleSelectPaper(p.id);
                          }}
                        />
                        <span className="truncate" title={p.filename}>
                          {p.title ? `${p.title} (${p.filename})` : p.filename}
                        </span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-gray-500">No notes yet.</p>
                )
              )}
            </section>

            {/* Insights Section */}
            <section>
              <button
                className="font-medium mb-2 flex items-center gap-1 bg-transparent border-0 p-0 hover:bg-transparent focus:outline-none"
                onClick={() => toggleCollapse("insights")}
              >
                <span>{collapsed.insights ? "▶" : "▼"}</span> Insights ({filteredInsights.length})
              </button>
              {!collapsed.insights && (
                filteredInsights && filteredInsights.length > 0 ? (
                  <ul className="space-y-1">
                    {filteredInsights.map((ins) => (
                      <li
                        key={ins.id}
                        className="flex items-center gap-2 truncate"
                        title={ins.title ?? ins.text.slice(0, 50)}
                      >
                        <input
                          type="checkbox"
                          checked={selectedInsights.has(ins.id)}
                          onChange={() => toggleSelectInsight(ins.id)}
                        />
                        <span
                          className="truncate flex-1 cursor-pointer hover:underline"
                          onClick={(e) => {
                            e.stopPropagation();
                            setModalInsight(ins);
                          }}
                        >
                          {ins.title || ins.text.slice(0, 50)}
                        </span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-gray-500">No insights yet.</p>
                )
              )}
            </section>
          </div>
        </div>

        {/* Collection management */}
        {/* Create collection */}
        <section>
          <h3 className="font-medium mb-1">Create Collection</h3>
          <div className="flex gap-2">
            <input
              ref={newCollInputRef}
              className="flex-1 border rounded px-2 py-1 bg-primaryBg text-defaultText border-primaryBg shadow-inner"
              value={newCollName}
              onChange={(e) => setNewCollName(e.target.value)}
            />
            <button
              className="bg-buttonBg text-defaultText px-3 py-1 rounded disabled:opacity-50 border border-primaryBg"
              disabled={!newCollName.trim() || createCollectionMut.isPending}
              onClick={() => createCollectionMut.mutate(newCollName.trim())}
            >
              Create
            </button>
          </div>
        </section>
      </div> {/* end background card */}

      {/* ---------------------------------------------------------------- */}
      {/* Debug utilities                                                */}
      {/* ---------------------------------------------------------------- */}

      <div className="bg-secondaryBg rounded-lg p-8 mt-6 space-y-4">
        <h3 className="font-medium">Debug</h3>
        {clearDbConfirm ? (
          <div className="flex items-center gap-2">
            <button
              className="px-3 py-1 rounded bg-[#db363c]/50 hover:bg-[#db363c] text-xs"
              disabled={clearDbMutation.isPending}
              onClick={() => clearDbMutation.mutate()}
            >
              Confirm Clear DB
            </button>
            <button
              className="w-7 h-7 p-0 flex items-center justify-center rounded hover:bg-white/10"
              onClick={() => setClearDbConfirm(false)}
            >
              <CloseIcon className="w-4 h-4" />
            </button>
          </div>
        ) : (
          <button
            className="px-3 py-1 rounded bg-buttonBg text-defaultText disabled:opacity-50"
            disabled={clearDbMutation.isPending}
            onClick={() => setClearDbConfirm(true)}
          >
            Clear Database
          </button>
        )}
      </div>
    </div>
  );
} 