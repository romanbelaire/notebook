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
} from "../api";
import type { Paper, Collection, Insight } from "../api";
import PlusIcon from "../assets/plus.svg?react";
import TrashIcon from "../assets/trash.svg?react";
import CloseIcon from "../assets/close.svg?react";
import MagnifyIcon from "../assets/magnify.svg?react";
import BookIcon from "../assets/book.svg?react";
import PencilIcon from "../assets/pencil.svg?react";
import PdfModal from "./PdfModal";
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
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCollId, setSelectedCollId] = useState<number | null>(null);
  const [showCollDropdown, setShowCollDropdown] = useState(false);
  const [editingCollName, setEditingCollName] = useState(false);
  const [newCollectionName, setNewCollectionName] = useState("");
  const [deleteConfirmColl, setDeleteConfirmColl] = useState(false);
  const dropdownRef = useRef<HTMLDivElement | null>(null);

  // Modal state for creating new collection
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [modalCollectionName, setModalCollectionName] = useState("");

  // Modal setter for insights
  const setModalInsight = useInsightsStore((s) => s.setModalInsight);

  // PDF viewer state
  const [pdfViewer, setPdfViewer] = useState<{ filename: string; page?: number } | null>(null);

  // Focus management: if ChatWindow requested focusing this input, open modal instead
  const focusNewCollection = useUIStore((s) => s.focusNewCollection);
  const clearFocusNewCollection = useUIStore((s) => s.clearFocusNewCollection);

  useEffect(() => {
    if (focusNewCollection) {
      setShowCreateModal(true);
      clearFocusNewCollection();
    }
  }, [focusNewCollection, clearFocusNewCollection]);

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

  // Get collection contents for dedicated section
  const collectionPapers = currentCollection && currentCollection.papers ? currentCollection.papers : [];
  const collectionPaperFiles = collectionPapers.filter((p) => !/\.(md|markdown|txt|docx)$/i.test(p.filename));
  const collectionNoteFiles = collectionPapers.filter((p) => /\.(md|markdown|txt|docx)$/i.test(p.filename));

  // Apply search filtering only (no collection filtering for main sections when collection is selected)
  const filteredPaperFiles = paperFiles
    .filter(p => matchesSearch(p.title) || matchesSearch(p.filename));
  
  const filteredNoteFiles = noteFiles
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
    },
  });

  // Create collection and add selected papers
  const createCollectionAndAddMut = useMutation<Collection, Error, string>({
    mutationFn: apiCreateCollection,
    onSuccess: (newCollection) => {
      queryClient.invalidateQueries({ queryKey: ["collections"] });
      setModalCollectionName("");
      setShowCreateModal(false);
      
      // If there are selected papers, add them to the new collection
      if (selectedPapers.size > 0) {
        addMut.mutate({ 
          collId: newCollection.id, 
          paperIds: Array.from(selectedPapers) 
        });
      }
    },
  });

  const addMut = useMutation<void, Error, { collId: number; paperIds: number[] }>({
    mutationFn: ({ collId, paperIds }) => apiAddToCollection(collId, paperIds),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["collections"] });
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
                  disabled={selectedPapers.size === 0 || addMut.isPending}
                  onClick={() => setShowAddMenu((prev) => !prev)}
                >
                  <PlusIcon className="w-4 h-4" />
                </button>
                {showAddMenu && (
                  <ul className="absolute mt-1 right-0 bg-primaryBg border border-trim/40 rounded shadow-lg z-30 min-w-[10rem]">
                    <li key="add-new">
                      <button
                        className="w-full text-left px-3 py-1 hover:bg-trim/20 border-b border-trim/20"
                        onClick={() => {
                          setShowAddMenu(false);
                          setShowCreateModal(true);
                        }}
                      >
                        + Add New Collection
                      </button>
                    </li>
                    {collections && collections.map((c) => (
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

              {/* PDF Viewer Button */}
              <button
                className="bg-buttonBg text-defaultText px-3 py-1 rounded disabled:opacity-50"
                disabled={(() => {
                  // Find first selected PDF
                  const selectedPdfIds = Array.from(selectedPapers).filter(id => {
                    const paper = paperFiles.find(p => p.id === id);
                    return paper && paper.filename.toLowerCase().endsWith('.pdf');
                  });
                  return selectedPdfIds.length === 0;
                })()}
                onClick={() => {
                  // Find first selected PDF and open it
                  const selectedPdfIds = Array.from(selectedPapers).filter(id => {
                    const paper = paperFiles.find(p => p.id === id);
                    return paper && paper.filename.toLowerCase().endsWith('.pdf');
                  });
                  if (selectedPdfIds.length > 0) {
                    const firstPdfId = selectedPdfIds[0];
                    const paper = paperFiles.find(p => p.id === firstPdfId);
                    if (paper) {
                      setPdfViewer({ filename: paper.filename });
                    }
                  }
                }}
                title="View PDF"
              >
                <MagnifyIcon className="w-5 h-5" />
              </button>
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

            {/* Collection Contents Section - shown only when viewing a collection */}
            {selectedCollId && currentCollection && (
              <section>
                <h3 className="font-medium mb-2 text-lg border-b border-trim/20 pb-1">
                  Collection Contents ({collectionPapers.length} documents)
                </h3>
                {collectionPapers.length > 0 ? (
                  <div className="space-y-3">
                    {/* Papers in Collection */}
                    {collectionPaperFiles.length > 0 && (
                      <div>
                        <h4 className="font-medium text-sm text-gray-400 mb-1">Papers ({collectionPaperFiles.length})</h4>
                        <ul className="space-y-1 ml-2">
                          {collectionPaperFiles.map((p) => (
                            <li key={p.id} className="flex items-center gap-2">
                              <input 
                                type="checkbox" 
                                checked={selectedPapers.has(p.id)} 
                                onChange={() => {
                                  // Unchecking removes from collection
                                  if (selectedPapers.has(p.id)) {
                                    removeFromCollMutation.mutate({ 
                                      collId: selectedCollId, 
                                      paperIds: [p.id] 
                                    });
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
                      </div>
                    )}
                    
                    {/* Notes in Collection */}
                    {collectionNoteFiles.length > 0 && (
                      <div>
                        <h4 className="font-medium text-sm text-gray-400 mb-1">Notes ({collectionNoteFiles.length})</h4>
                        <ul className="space-y-1 ml-2">
                          {collectionNoteFiles.map((p) => (
                            <li key={p.id} className="flex items-center gap-2">
                              <input 
                                type="checkbox" 
                                checked={selectedPapers.has(p.id)} 
                                onChange={() => {
                                  // Unchecking removes from collection
                                  if (selectedPapers.has(p.id)) {
                                    removeFromCollMutation.mutate({ 
                                      collId: selectedCollId, 
                                      paperIds: [p.id] 
                                    });
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
                      </div>
                    )}
                  </div>
                ) : (
                  <p className="text-sm text-gray-500">This collection is empty.</p>
                )}
              </section>
            )}

            {/* Select All Section */}
            <section>
              <div className="flex items-center gap-2 mb-4 pb-2 border-b border-trim/20">
                <input
                  type="checkbox"
                  checked={(() => {
                    const allVisibleDocuments = [
                      ...filteredPaperFiles.map(p => p.id),
                      ...filteredNoteFiles.map(p => p.id)
                    ];
                    const allVisibleInsights = filteredInsights.map(i => i.id);
                    
                    // If no documents to select, return false
                    if (allVisibleDocuments.length === 0 && allVisibleInsights.length === 0) {
                      return false;
                    }
                    
                    // Check if all are selected
                    const allPapersSelected = allVisibleDocuments.every(id => selectedPapers.has(id));
                    const allInsightsSelected = allVisibleInsights.every(id => selectedInsights.has(id));
                    
                    return allPapersSelected && allInsightsSelected;
                  })()}
                  ref={(el) => {
                    if (el) {
                      const allVisibleDocuments = [
                        ...filteredPaperFiles.map(p => p.id),
                        ...filteredNoteFiles.map(p => p.id)
                      ];
                      const allVisibleInsights = filteredInsights.map(i => i.id);
                      
                      const selectedPaperCount = allVisibleDocuments.filter(id => selectedPapers.has(id)).length;
                      const selectedInsightCount = allVisibleInsights.filter(id => selectedInsights.has(id)).length;
                      const totalSelected = selectedPaperCount + selectedInsightCount;
                      const totalVisible = allVisibleDocuments.length + allVisibleInsights.length;
                      
                      // Set indeterminate state if partially selected
                      el.indeterminate = totalSelected > 0 && totalSelected < totalVisible;
                    }
                  }}
                  onChange={() => {
                    const allVisibleDocuments = [
                      ...filteredPaperFiles.map(p => p.id),
                      ...filteredNoteFiles.map(p => p.id)
                    ];
                    const allVisibleInsights = filteredInsights.map(i => i.id);
                    
                    // Check current state
                    const allPapersSelected = allVisibleDocuments.every(id => selectedPapers.has(id));
                    const allInsightsSelected = allVisibleInsights.every(id => selectedInsights.has(id));
                    const allSelected = allPapersSelected && allInsightsSelected;
                    
                    if (allSelected) {
                      // Unselect all
                      setSelectedPapers(new Set());
                      setSelectedInsights(new Set());
                    } else {
                      // Select all
                      setSelectedPapers(new Set(allVisibleDocuments));
                      setSelectedInsights(new Set(allVisibleInsights));
                    }
                  }}
                />
                <span className="font-medium text-sm">
                  Select All ({filteredPaperFiles.length + filteredNoteFiles.length + filteredInsights.length} items)
                </span>
              </div>
            </section>

            {/* Papers Section - All Papers */}
            <section>
              <button
                className="font-medium mb-2 flex items-center gap-1 bg-transparent border-0 p-0 hover:bg-transparent focus:outline-none"
                onClick={() => toggleCollapse("papers")}
              >
                <span>{collapsed.papers ? "▶" : "▼"}</span> {selectedCollId ? "All Papers" : "Papers"} ({filteredPaperFiles.length})
              </button>
              {!collapsed.papers && (
                filteredPaperFiles.length > 0 ? (
                  <ul className="space-y-1">
                    {filteredPaperFiles.map((p) => (
                      <li key={p.id} className="flex items-center gap-2">
                        <input 
                          type="checkbox" 
                          checked={selectedPapers.has(p.id)} 
                          onChange={() => toggleSelectPaper(p.id)} 
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

            {/* Notes Section - All Notes */}
            <section>
              <button
                className="font-medium mb-2 flex items-center gap-1 bg-transparent border-0 p-0 hover:bg-transparent focus:outline-none"
                onClick={() => toggleCollapse("notes")}
              >
                <span>{collapsed.notes ? "▶" : "▼"}</span> {selectedCollId ? "All Notes" : "Notes"} ({filteredNoteFiles.length})
              </button>
              {!collapsed.notes && (
                filteredNoteFiles.length > 0 ? (
                  <ul className="space-y-1">
                    {filteredNoteFiles.map((p) => (
                      <li key={p.id} className="flex items-center gap-2">
                        <input 
                          type="checkbox" 
                          checked={selectedPapers.has(p.id)} 
                          onChange={() => toggleSelectPaper(p.id)}
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
      </div> {/* end background card */}

      {/* PDF Modal */}
      {pdfViewer && (
        <PdfModal
          filename={pdfViewer.filename}
          initialPage={pdfViewer.page}
          onClose={() => setPdfViewer(null)}
        />
      )}

      {/* Create Collection Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-primaryBg rounded-lg p-6 w-96 max-w-[90vw]">
            <h3 className="font-medium mb-4 text-lg">Create New Collection</h3>
            <div className="space-y-4">
              <input
                type="text"
                placeholder="Collection name..."
                className="w-full border rounded px-3 py-2 bg-secondaryBg text-defaultText border-trim/40 focus:outline-none focus:border-trim"
                value={modalCollectionName}
                onChange={(e) => setModalCollectionName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && modalCollectionName.trim()) {
                    createCollectionAndAddMut.mutate(modalCollectionName.trim());
                  } else if (e.key === 'Escape') {
                    setShowCreateModal(false);
                    setModalCollectionName("");
                  }
                }}
                autoFocus
              />
              <div className="flex gap-2 justify-end">
                <button
                  className="px-4 py-2 rounded bg-buttonBg text-defaultText hover:bg-buttonBg/80"
                  onClick={() => {
                    setShowCreateModal(false);
                    setModalCollectionName("");
                  }}
                  disabled={createCollectionAndAddMut.isPending}
                >
                  Cancel
                </button>
                <button
                  className="px-4 py-2 rounded bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
                  disabled={!modalCollectionName.trim() || createCollectionAndAddMut.isPending}
                  onClick={() => createCollectionAndAddMut.mutate(modalCollectionName.trim())}
                >
                  {createCollectionAndAddMut.isPending ? "Creating..." : "Create"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}


    </div>
  );
} 