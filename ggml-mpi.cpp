#include "ggml-mpi.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <atomic>
#include <deque>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define UNUSED GGML_UNUSED

static std::atomic<bool> have_init {false};

static void* send_buffer;

struct inference_run {
    inference_run(MPI_Request pRequest, bool b) {
        logits_receive_request = pRequest;
        received = b;
    }

    MPI_Request logits_receive_request;
    bool received;
};

struct ggml_mpi_context {
    int rank;
    size_t size;
    MPI_Comm comm;
    int layer_start;
    int layer_end;
    MPI_Status status;

    std::string name;
    std::vector<ggml_backend_t> backends;
    bool remote;
    void* send_buffer;
    int trans_id;
    int recv_trans_id;
    std::deque<inference_run> requests{};
};

void ggml_mpi_backend_init(void) {
    if (!have_init) {
        int ret;

        GGML_ASSERT(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &ret) == MPI_SUCCESS);
        have_init = true;
        const int buffer_size = 128 * 1024 * 1024 * 8;
        send_buffer = calloc(1, buffer_size); // 128MB buffer
        fprintf(stderr, "BUFFER ATTACH RETCODE=%d\n", MPI_Buffer_attach(send_buffer, buffer_size));
    }
}

void ggml_mpi_sync_pipelined(
        struct ggml_mpi_context *   ctx_mpi,
        void * val,
        int count,
        MPI_Datatype datatype,
        int tag
);

void ggml_mpi_backend_free(void) {
    MPI_Finalize();
}

struct ggml_mpi_context * ggml_mpi_init(void) {


    ggml_mpi_backend_init();


    auto * ctx = new ggml_mpi_context;

    int size = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &ctx->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    ctx->size = size;
    ctx->comm = MPI_COMM_WORLD;
    ctx->remote = false;

    ctx->send_buffer = send_buffer;

    return ctx;
}

struct ggml_mpi_context * ggml_mpi_split_comm(struct ggml_mpi_context * ctx, int color, int key) {
    fprintf(stderr, "SPLITTING COMM WITH COLOR %d AND KEY %d ON RANK %d\n", color, key, ctx->rank);
    auto * newCtx = ggml_mpi_init();
    int ret = MPI_Comm_split(ctx->comm, (color >= 0) ? color : MPI_UNDEFINED, key, &newCtx->comm);
    if (ret != MPI_SUCCESS) {
        fprintf(stderr, "SPLIT RETURNED %d\n", ret);
        GGML_ASSERT(false);
    }
    if (newCtx->comm != MPI_COMM_NULL) {
        fprintf(stderr, "CREATED NON NULL COMM ON RANK %d\n", ctx->rank);
        MPI_Comm_rank(newCtx->comm, &newCtx->rank);
        int size = 0;
        MPI_Comm_size(newCtx->comm, &size);
        newCtx->size = size;
        fprintf(stderr, "NON NULL COMM SIZE %zu, RANK %d\n", newCtx->size, newCtx->rank);

    } else {
        fprintf(stderr, "CREATED NULL COMM ON RANK %d\n", ctx->rank);
        newCtx->rank = -1;
        newCtx->size = 0;
        newCtx->remote = true;
    }
    return newCtx;
}

void ggml_mpi_free(struct ggml_mpi_context * ctx) {
    if(ctx->comm == MPI_COMM_NULL) {
        return;
    }

    if (ctx->comm == nullptr) {
        return;
    }

    MPI_Comm_free(&(ctx->comm));
    free(ctx);
}

int ggml_mpi_rank(struct ggml_mpi_context * ctx) {
    return ctx->rank;
}

size_t ggml_mpi_size(struct ggml_mpi_context * ctx) {
    return ctx->size;
}

int ggml_mpi_next_node(struct ggml_mpi_context * ctx_mpi) {
    return (ctx_mpi->rank + 1) % (int)ctx_mpi->size;
}

int ggml_mpi_prev_node(struct ggml_mpi_context * ctx_mpi) {
    int temp = (ctx_mpi->rank - 1);
    return (temp >= 0) ? temp : (int)ctx_mpi->size - 1;
}

void ggml_mpi_sync_pipelined(
        struct ggml_mpi_context *   ctx_mpi,
        void * val,
        int count,
        MPI_Datatype datatype,
        int tag
) {
    if(ctx_mpi->comm == MPI_COMM_NULL) {
        return;
    }

    if (ctx_mpi->comm == nullptr) {
        return;
    }

//    printf("Rank %d sync pipelined with tag %d\n", ctx_mpi->rank, tag);


    if (ctx_mpi->rank != 0) {
        MPI_Recv(val, count, datatype, ggml_mpi_prev_node(ctx_mpi), tag, ctx_mpi->comm, MPI_STATUS_IGNORE);
    }
    if(ctx_mpi->rank < (int)ctx_mpi->size - 1) {
        GGML_ASSERT(ctx_mpi->send_buffer != nullptr);
        GGML_ASSERT(val != nullptr || count == 0);
        GGML_ASSERT(count < 128*1024*1024);

        const int retval = MPI_Bsend(val, count, datatype, ggml_mpi_next_node(ctx_mpi), tag, ctx_mpi->comm);
        GGML_ASSERT(retval == MPI_SUCCESS);

    }
}

void ggml_mpi_barrier(struct ggml_mpi_context * ctx_mpi) {
    if(ctx_mpi->comm == MPI_COMM_NULL) {
        return;
    }

    if (ctx_mpi->comm == nullptr) {
        return;
    }
    MPI_Barrier(ctx_mpi->comm);
}

void ggml_mpi_probe(struct ggml_mpi_context * ctx_mpi, int src, int tag) {
    if(ctx_mpi->comm == MPI_COMM_NULL) {
        return;
    }

    if (ctx_mpi->comm == nullptr) {
        return;
    }
    MPI_Probe((src >= 0) ? src : MPI_ANY_SOURCE, (tag >= 0) ? tag : MPI_ANY_TAG, ctx_mpi->comm, &(ctx_mpi->status));
}

int ggml_mpi_iprobe(struct ggml_mpi_context * ctx_mpi, int src, int tag) {
    if(ctx_mpi->comm == MPI_COMM_NULL) {
        return 0;
    }

    if (ctx_mpi->comm == nullptr) {
        return 0;
    }

    int ret;
    MPI_Iprobe((src >= 0) ? src : MPI_ANY_SOURCE, (tag >= 0) ? tag : MPI_ANY_TAG, ctx_mpi->comm, &ret, &(ctx_mpi->status));
    return ret;
}



int ggml_mpi_status_tag(struct ggml_mpi_context * ctx_mpi) {
    return ctx_mpi->status.MPI_TAG;
}

int ggml_mpi_status_count_int32(struct ggml_mpi_context * ctx_mpi) {
    int32_t count;
    MPI_Get_count(&ctx_mpi->status, MPI_INT32_T, &count);
    return count;
}

void ggml_mpi_eval_init(
        struct ggml_mpi_context *   ctx_mpi,
                int32_t         *   n_tokens,
                int32_t         **  pos,
                int32_t         **  n_seq_ids,
                int32_t         *** seq_id,
                int8_t          **  logits,
                uint32_t            n_seq_max,
                size_t          *   batch_id) {


    if(ctx_mpi->comm == MPI_COMM_NULL) {
        return;
    }
    int32_t old_n_tokens = *n_tokens;


    ggml_mpi_sync_pipelined(ctx_mpi, batch_id, 1, MPI_INT, GGML_MPI_BATCH_ID);


    ggml_mpi_sync_pipelined(ctx_mpi, n_tokens, 1, MPI_INT, GGML_MPI_N_TOKENS);
    auto* temp_logits = (int8_t*) calloc(*n_tokens, sizeof(int8_t));

    if (ctx_mpi->rank == 0 && *logits != NULL) {
        ggml_mpi_sync_pipelined(ctx_mpi, *logits, *n_tokens, MPI_INT8_T, GGML_MPI_BATCH_LOGITS);
    } else {
        ggml_mpi_sync_pipelined(ctx_mpi, temp_logits, *n_tokens, MPI_INT8_T, GGML_MPI_BATCH_LOGITS);
    }







    if (ctx_mpi->rank != 0) {
        bool should_set_batch_logits = false;
        for (int i = 0; i < *n_tokens; i++) {
            if (temp_logits[i]) {
                should_set_batch_logits = true;
                break;
            }
        }
        if (should_set_batch_logits) {
            if (*logits != NULL) {
                free(*logits);
                *logits = NULL;
            }
            *logits = temp_logits;
        } else {
            if (*logits != NULL) {
                free(*logits);
                *logits = NULL;
            }
            free(temp_logits);
        }
    } else {
        free(temp_logits);
    }

    // For now, we assume that the pos, seq_ids, tokens, etc have been
    // pre-allocated for the largest possible sizes, even on worker nodes.
    //if (old_n_tokens != *n_tokens) {
    //    *pos = realloc(*pos, *n_tokens * sizeof(int32_t));
    //    *n_seq_ids = realloc(*n_seq_ids, *n_tokens * sizeof(int32_t ));
    //    *tokens = realloc(*tokens, *n_tokens * sizeof(int32_t ));
    //}


    ggml_mpi_sync_pipelined(ctx_mpi, *n_seq_ids, *n_tokens, MPI_INT32_T, GGML_MPI_N_SEQ_IDS);

    // We need to know the total number of sequence
    // ids, so we count them all up
    int32_t total_n_seq_ids = 0;
    for (int32_t i = 0; i < *n_tokens; i++) {
        total_n_seq_ids += (*n_seq_ids)[i];
    }

    // MPI can't chase the pointers for multidimensional arrays, so we flatten them first
    // for transit
    auto * flattened_seq_ids = static_cast<int32_t *>(calloc(total_n_seq_ids, sizeof(int32_t)));

    int32_t current_index = 0;

    // Only rank 0 needs to flatten since the others don't have the real seq_id
    if (ctx_mpi->rank == 0) {
        for (int32_t i = 0; i < *n_tokens; i++) {
            for (int32_t j = 0; j < (*n_seq_ids)[i]; j++) {
                flattened_seq_ids[current_index] = (*seq_id)[i][j];
                current_index++;
            }
        }
    }



    ggml_mpi_sync_pipelined(ctx_mpi, *pos, *n_tokens, MPI_INT32_T, GGML_MPI_POS);
    ggml_mpi_sync_pipelined(ctx_mpi, flattened_seq_ids, total_n_seq_ids, MPI_INT32_T, GGML_MPI_SEQ_IDS);

    current_index = 0;
    if (ctx_mpi->rank != 0) {
        for (int32_t i = 0; i < *n_tokens; i++) {
            if ((*seq_id)[i] != nullptr) {
                free((*seq_id)[i]);
            }
            (*seq_id)[i] = new int32_t[(*n_seq_ids)[i]];
            for (int32_t j = 0; j < (*n_seq_ids)[i]; j++) {
                (*seq_id)[i][j] = flattened_seq_ids[current_index];
                current_index++;
            }

        }
    }
    free(flattened_seq_ids);
}


void ggml_mpi_sync_int32_t(
        struct ggml_mpi_context * ctx_mpi,
                        int32_t * val,
                        int root
) {
    if(ctx_mpi->comm == MPI_COMM_NULL) {
        return;
    }

    if (ctx_mpi->comm == nullptr) {
        return;
    }

//    fprintf(stderr, "SYNCING INTEGER WITH ROOT %d, SIZE %zu, AND RANK %d\n", root, ctx_mpi->size, ctx_mpi->rank);
    MPI_Bcast(val, 1, MPI_INT32_T, root, ctx_mpi->comm);
//    fprintf(stderr, "FINISHED SYNCING INTEGER WITH ROOT %d, SIZE %zu, AND RANK %d\n", root, ctx_mpi->size, ctx_mpi->rank);

}

void ggml_mpi_sync_bool(
        struct ggml_mpi_context * ctx_mpi,
        bool * val,
        int root
) {
    if(ctx_mpi->comm == MPI_COMM_NULL) {
        return;
    }

    if (ctx_mpi->comm == nullptr) {
        return;
    }

//    fprintf(stderr, "SYNCING BOOL WITH ROOT %d, SIZE %zu, AND RANK %d\n", root, ctx_mpi->size, ctx_mpi->rank);


    MPI_Bcast(val, 1, MPI_CXX_BOOL, root, ctx_mpi->comm);

//    fprintf(stderr, "FINISHED SYNCING BOOL WITH ROOT %d, SIZE %zu, AND RANK %d\n", root, ctx_mpi->size, ctx_mpi->rank);

}

void ggml_mpi_sync_float(
        struct ggml_mpi_context * ctx_mpi,
        float * val,
        int root
) {
    if(ctx_mpi->comm == MPI_COMM_NULL) {
        return;
    }

    if (ctx_mpi->comm == nullptr) {
        return;
    }
//    printf("Rank %d sync float\n", ctx_mpi->rank);

//    fprintf(stderr, "SYNCING FLOAT WITH ROOT %d, SIZE %zu, AND RANK %d\n", root, ctx_mpi->size, ctx_mpi->rank);

    MPI_Bcast(val, 1, MPI_FLOAT, root, ctx_mpi->comm);

//    fprintf(stderr, "FINISHED SYNCING FLOAT WITH ROOT %d, SIZE %zu, AND RANK %d\n", root, ctx_mpi->size, ctx_mpi->rank);

}

void ggml_mpi_sync_ints_pipelined(
        struct ggml_mpi_context * ctx_mpi,
        int32_t * vals,
        int count,
        int tag
) {
    if(ctx_mpi->comm == MPI_COMM_NULL) {
        return;
    }

    if (ctx_mpi->comm == nullptr) {
        return;
    }
    ggml_mpi_sync_pipelined(ctx_mpi, vals, count, MPI_INT32_T, tag);
    int old_trans = ctx_mpi->trans_id;
    ggml_mpi_sync_pipelined(ctx_mpi, &ctx_mpi->trans_id, 1, MPI_INT32_T, GGML_MPI_TRANS_ID);
    ctx_mpi->recv_trans_id = ctx_mpi->trans_id;
    ctx_mpi->trans_id = old_trans;
}

void ggml_mpi_sync_pipelined_back(
        struct ggml_mpi_context *   ctx_mpi,
        void * val,
        int count,
        MPI_Datatype datatype,
        int tag
) {
    if(ctx_mpi->comm == MPI_COMM_NULL) {
        return;
    }

    //printf("Rank %d sync pipelined\n", ctx_mpi->rank);


    if (ctx_mpi->rank != 0) {
        MPI_Recv(val, count, datatype, ggml_mpi_next_node(ctx_mpi), tag, ctx_mpi->comm, MPI_STATUS_IGNORE);
    }
    if(ctx_mpi->rank != 1) {
        const int retval = MPI_Bsend(val, count, datatype, ggml_mpi_prev_node(ctx_mpi), tag, ctx_mpi->comm);
        GGML_ASSERT(retval == MPI_SUCCESS);

    }
}

void ggml_mpi_sync_ints_pipelined_back(
        struct ggml_mpi_context * ctx_mpi,
        int32_t * vals,
        int count,
        int tag
) {
    ggml_mpi_sync_pipelined_back(ctx_mpi, vals, count, MPI_INT32_T, tag);
//    int old_trans = ctx_mpi->trans_id;
//    ggml_mpi_sync_pipelined_back(ctx_mpi, &ctx_mpi->trans_id, 1, MPI_INT32_T, GGML_MPI_TRANS_ID);
//    ctx_mpi->recv_trans_id = ctx_mpi->trans_id;
//    ctx_mpi->trans_id = old_trans;
}

static void ggml_mpi_tensor_send(const struct ggml_tensor * t, const void* data, int mpi_rank_dst, MPI_Comm comm) {
    MPI_Datatype mpi_type;

//    fprintf(stderr, "Type: %d\n", t->type);

    switch (t->type) {
        case GGML_TYPE_I32: mpi_type = MPI_INT32_T; break;
        case GGML_TYPE_F32: mpi_type = MPI_FLOAT;   break;
        case GGML_TYPE_F16: mpi_type = MPI_INT16_T;   break;
        default: GGML_ASSERT(false && "not implemented");
    }
    int rank;
    MPI_Comm_rank(comm, &rank);
//    fprintf(stderr, "Sending tensor %s (buffer %s) from %d to %d\n", t->name, ggml_backend_buffer_name(t->buffer), rank, mpi_rank_dst);

    GGML_ASSERT(rank != mpi_rank_dst);

    const int retval = MPI_Bsend(data, (int)ggml_nelements(t), mpi_type, mpi_rank_dst, 0, comm);
    GGML_ASSERT(retval == MPI_SUCCESS);

}
static void ggml_mpi_tensor_send(const struct ggml_tensor * t, int mpi_rank_dst, MPI_Comm comm) {
    ggml_mpi_tensor_send(t, t->data, mpi_rank_dst, comm);
}

static void ggml_mpi_tensor_recv(const struct ggml_tensor * t, void * data, int mpi_rank_src, MPI_Comm comm) {
    MPI_Datatype mpi_type;

    switch (t->type) {
        case GGML_TYPE_I32: mpi_type = MPI_INT32_T; break;
        case GGML_TYPE_F32: mpi_type = MPI_FLOAT;   break;
        default: GGML_ASSERT(false && "not implemented");
    }

    MPI_Status status; UNUSED(status);
//    fprintf(stderr, "%s: tensor receive == null: %d\n", __func__, t->data == NULL);
    int rank;
    MPI_Comm_rank(comm, &rank);
//    fprintf(stderr, "Receiving tensor %s (buffer %s) from %d at %d\n", t->name, ggml_backend_buffer_name(t->buffer), mpi_rank_src, rank);

    GGML_ASSERT(rank != mpi_rank_src);

    const int retval = MPI_Recv(data, (int)ggml_nelements(t), mpi_type, mpi_rank_src, 0, comm, &status);
    GGML_ASSERT(retval == MPI_SUCCESS);
//    fprintf(stderr, "Done: Received tensor %s (buffer %s) from %d at %d\n", t->name, ggml_backend_buffer_name(t->buffer), mpi_rank_src, rank);

}

static void ggml_mpi_tensor_recv(struct ggml_tensor * t, int mpi_rank_src, MPI_Comm comm) {
    ggml_mpi_tensor_recv(t, t->data, mpi_rank_src, comm);
}

static void ggml_mpi_tensor_recv_async(ggml_mpi_context * ctx_mpi, const struct ggml_tensor * t, void * data, int mpi_rank_src, MPI_Comm comm) {
    MPI_Datatype mpi_type;

    switch (t->type) {
        case GGML_TYPE_I32: mpi_type = MPI_INT32_T; break;
        case GGML_TYPE_F32: mpi_type = MPI_FLOAT;   break;
        default: GGML_ASSERT(false && "not implemented");
    }

    MPI_Request request;
//    fprintf(stderr, "%s: tensor receive == null: %d\n", __func__, t->data == NULL);
    int rank;
    MPI_Comm_rank(comm, &rank);
//    fprintf(stderr, "Receiving tensor async %s (buffer %s) from %d at %d\n", t->name, ggml_backend_buffer_name(t->buffer), mpi_rank_src, rank);

    GGML_ASSERT(rank != mpi_rank_src);

    const int retval = MPI_Irecv(data, (int)ggml_nelements(t), mpi_type, mpi_rank_src, 0, comm, &request);
    GGML_ASSERT(retval == MPI_SUCCESS);
    ctx_mpi->requests.emplace_back(request, false);
}

uint16_t** ggml_mpi_split_range(
    struct ggml_mpi_context * ctx_mpi,
    uint16_t start,
    uint16_t end,
    const float node_weights[]
) {
    // Splits the range given by start and end
    // over the available nodes. This implementation
    // assumes that node 0 handles the final part of the range
    // while node 1 handles the beginning, to form a ring pipeline

    uint16_t range_length = end - start + 1;
    auto ** ranges = (uint16_t**) malloc(sizeof(uint16_t*) * ctx_mpi->size);
    for (size_t i = 0; i < ctx_mpi->size; i++) {
        ranges[i] = (uint16_t*) malloc(sizeof(uint16_t) * 2);
    }
    uint16_t next_layer = 0;
    for (size_t i=0; i < ctx_mpi->size; i++) {
        ranges[i][0] = next_layer;
        ranges[i][1] = MIN(end, ranges[i][0] + (node_weights[i] * range_length) + start);
        next_layer = ranges[i][1]+1;
    }

//    ranges[0][0] = next_layer;
//    ranges[0][1] = MIN(end, next_layer + (node_weights[0] * range_length) + start);
    return ranges;

}

// BACKEND V2

struct ggml_backend_mpi_buffer_context {
    ggml_backend_buffer_t wrapped_buffer;
    ggml_mpi_context * ctx_mpi;
    bool empty;
};

struct ggml_backend_mpi_buffer_type_context {
    std::string name;
    ggml_backend_buffer_type_t wrapped_buffer_type;
    ggml_mpi_context * ctx_mpi;
    bool empty;
};

MPI_Comm ggml_backend_mpi_get_comm(ggml_backend_t backend) {
    auto * ctx = (ggml_mpi_context *) backend->context;

    return ctx->comm;
}

int ggml_backend_mpi_rank(ggml_backend_t backend) {
    auto * ctx = (ggml_mpi_context *) backend->context;
    return ctx->rank;
}

int ggml_backend_mpi_local_rank(ggml_backend_t backend) {
    int rank;
    int ret = MPI_Comm_rank(ggml_backend_mpi_get_comm(backend), &rank);
    GGML_ASSERT(ret == MPI_SUCCESS);
    return rank;
}

static const char * ggml_backend_mpi_name(ggml_backend_t backend) {
    int rank;
    MPI_Comm_rank(ggml_backend_mpi_get_comm(backend), &rank);

    return strdup(("MPI(Rank " + std::to_string(ggml_backend_mpi_rank(backend)) + ", local rank " + std::to_string(ggml_backend_mpi_local_rank(backend)) + ", comm size " + std::to_string(((ggml_mpi_context*)backend->context)->size) + ")").c_str());
}

int ggml_backend_mpi_buffer_type_rank(ggml_backend_buffer_type_t buft);

int ggml_backend_mpi_buffer_type_local_rank(ggml_backend_buffer_type_t buft);

GGML_CALL static const char * ggml_backend_mpi_buffer_type_name(ggml_backend_buffer_type_t buft) {
    auto * ctx = (ggml_backend_mpi_buffer_type_context *) buft->context;


    return strdup(
            (
                    ctx->name +
                    " Buffer Type(Rank " +
                    std::to_string(
                            ggml_backend_mpi_buffer_type_rank(buft)
                    ) +
                    ", local rank " +
                    std::to_string(ggml_backend_mpi_buffer_type_local_rank(buft)) +
                    ", comm_size: " +
                    std::to_string(((ggml_backend_mpi_buffer_type_context*)buft->context)->ctx_mpi->size) +
                    "):" +
                    std::string(
                            ctx->wrapped_buffer_type->iface.get_name(ctx->wrapped_buffer_type)
                    )
            ).c_str()
    );
}

MPI_Comm ggml_backend_mpi_buffer_type_get_comm(ggml_backend_buffer_type_t buft) {
    auto * buft_ctx = (ggml_backend_mpi_buffer_type_context *) buft->context;
    return buft_ctx->ctx_mpi->comm;

}

MPI_Comm ggml_backend_mpi_buffer_get_comm(ggml_backend_buffer_t buffer) {
    return ggml_backend_mpi_buffer_type_get_comm(buffer->buft);
}



int ggml_backend_mpi_buffer_local_rank(ggml_backend_buffer_t buffer) {
    if(ggml_backend_mpi_buffer_get_comm(buffer) == MPI_COMM_NULL) {
        return -1;
    }

    if (ggml_backend_mpi_buffer_get_comm(buffer) == nullptr) {
        return -1;
    }
    int rank;
    int ret = MPI_Comm_rank(ggml_backend_mpi_buffer_get_comm(buffer), &rank);
    GGML_ASSERT(ret == MPI_SUCCESS);
    return rank;
}

int ggml_backend_mpi_buffer_type_local_rank(ggml_backend_buffer_type_t buft) {
    if(ggml_backend_mpi_buffer_type_get_comm(buft) == MPI_COMM_NULL) {
        return -1;
    }

    if (ggml_backend_mpi_buffer_type_get_comm(buft) == nullptr) {
        return -1;
    }
    int rank;
    int ret = MPI_Comm_rank(ggml_backend_mpi_buffer_type_get_comm(buft), &rank);
    GGML_ASSERT(ret == MPI_SUCCESS);
    return rank;
}



int ggml_backend_mpi_buffer_rank(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer != nullptr);
    auto * ctx = (ggml_backend_mpi_buffer_context *) buffer->context;
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(ctx->ctx_mpi != nullptr);
    return ctx->ctx_mpi->rank;
}

ggml_mpi_context * ggml_backend_mpi_buffer_ctx(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer != nullptr);
    auto * ctx = (ggml_backend_mpi_buffer_context *) buffer->context;
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(ctx->ctx_mpi != nullptr);
    return ctx->ctx_mpi;
}

int ggml_backend_mpi_buffer_type_rank(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft->iface.get_name == ggml_backend_mpi_buffer_type_name);
    auto * ctx = (ggml_backend_mpi_buffer_type_context *) buft->context;
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(ctx->ctx_mpi != nullptr);
    return ctx->ctx_mpi->rank;
}

ggml_mpi_context * ggml_backend_mpi_buffer_type_ctx(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft != nullptr);
    auto * ctx = (ggml_backend_mpi_buffer_type_context *) buft->context;
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(ctx->ctx_mpi != nullptr);
    return ctx->ctx_mpi;
}



GGML_CALL static const char * ggml_backend_mpi_buffer_name(ggml_backend_buffer_t buffer);

ggml_backend_buffer_type_t ggml_backend_mpi_buffer_type_unwrap(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft != nullptr);
    auto * ctx = (ggml_backend_mpi_buffer_type_context *) buft->context;

    GGML_ASSERT(ctx != nullptr);

    ggml_backend_buffer_type_t wrapped_buffer_type = ctx->wrapped_buffer_type;
    return wrapped_buffer_type;

}

ggml_backend_buffer_t ggml_backend_mpi_buffer_unwrap(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer != nullptr);
//    fprintf(stderr, "Attempting unwrap of %s\n", ggml_backend_buffer_name(buffer));
//    if(buffer->iface.get_name != ggml_backend_mpi_buffer_name) {
//        return buffer;
//    }
    auto * ctx = (ggml_backend_mpi_buffer_context *) buffer->context;
    GGML_ASSERT(ctx != nullptr);
    ggml_backend_buffer_t wrapped_buffer = ctx->wrapped_buffer;
    GGML_ASSERT(wrapped_buffer != nullptr);
    wrapped_buffer->usage = buffer->usage;
    wrapped_buffer->size = buffer->size;
    if (wrapped_buffer->buft->iface.get_name == ggml_backend_mpi_buffer_type_name) {
        wrapped_buffer->buft = ggml_backend_mpi_buffer_type_unwrap(wrapped_buffer->buft);
    }
    return wrapped_buffer;

}




GGML_CALL static const char * ggml_backend_mpi_buffer_name(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(ggml_backend_mpi_buffer_unwrap(buffer) != nullptr && ggml_backend_mpi_buffer_unwrap(buffer)->iface.get_name != ggml_backend_mpi_buffer_name);

    return strdup(
            (

                    "MPI Buffer(Rank " +
                    std::to_string(ggml_backend_mpi_buffer_rank(buffer)) +
                    ", local rank " +
                    std::to_string(ggml_backend_mpi_buffer_local_rank(buffer)) +
                    ", comm_size: " +
                    std::to_string(((ggml_backend_mpi_buffer_context*)buffer->context)->ctx_mpi->size) +
                    "):" +
                    std::string(
                            ggml_backend_buffer_name(
                                    ggml_backend_mpi_buffer_unwrap(buffer)
                            )
                    )
            ).c_str()
    );
}

GGML_CALL void ggml_backend_mpi_buffer_type_copy_ctx(ggml_backend_buffer_type_t src, ggml_backend_buffer_type_t dst) {
    if (src->iface.get_name == ggml_backend_mpi_buffer_type_name) {
        ((ggml_backend_mpi_buffer_type_context *) dst->context)->ctx_mpi = ((ggml_backend_mpi_buffer_type_context *) src->context)->ctx_mpi;
    } else {
        GGML_ASSERT(!"Buffer type must be wrapped in ggml_backend_mpi_buffer_type_t");
    }
}

GGML_CALL void ggml_backend_mpi_buffer_copy_ctx(ggml_backend_buffer_t src, ggml_backend_buffer_t dst) {
    if (src->iface.get_name == ggml_backend_mpi_buffer_name) {
        *((ggml_backend_mpi_buffer_context *) dst->context)->ctx_mpi = *((ggml_backend_mpi_buffer_context *) src->context)->ctx_mpi;
        ggml_backend_mpi_buffer_type_copy_ctx(src->buft, dst->buft);
    } else {
        GGML_ASSERT(!"Buffer must be wrapped in ggml_backend_mpi_buffer_t");
    }
}

GGML_CALL void ggml_backend_mpi_buffer_copy_ctx_from_type(ggml_backend_buffer_type_t src, ggml_backend_buffer_t dst) {
    if (src->iface.get_name == ggml_backend_mpi_buffer_type_name) {
        ((ggml_backend_mpi_buffer_context *) dst->context)->ctx_mpi = ((ggml_backend_mpi_buffer_type_context *) src->context)->ctx_mpi;
        ggml_backend_mpi_buffer_type_copy_ctx(src, dst->buft);
    } else {
        GGML_ASSERT(!"Buffer must be wrapped in ggml_backend_mpi_buffer_t");
    }
}

GGML_CALL void ggml_backend_mpi_buffer_type_copy_ctx_from_backend(ggml_backend_buffer_type_t dst, ggml_backend_t src) {
    if (src->iface.get_name == ggml_backend_mpi_name) {
        ((ggml_backend_mpi_buffer_type_context *) dst->context)->ctx_mpi = ((ggml_mpi_context *) src->context);
    } else {
        GGML_ASSERT(!"Buffer must be wrapped in ggml_backend_mpi_buffer_t");
    }
}

GGML_CALL void ggml_backend_mpi_buffer_copy_ctx_from_backend(ggml_backend_buffer_t dst, ggml_backend_t src) {
    if (src->iface.get_name == ggml_backend_mpi_name) {
        ((ggml_backend_mpi_buffer_context *) dst->context)->ctx_mpi = ((ggml_mpi_context *) src->context);
        ggml_backend_mpi_buffer_type_copy_ctx_from_backend(dst->buft, src);
    } else {
        GGML_ASSERT(!"Buffer must be wrapped in ggml_backend_mpi_buffer_t");
    }
}

ggml_backend_buffer_type_t ggml_backend_mpi_buffer_type_set_wrapped_buffer_type(ggml_backend_buffer_type_t orig, ggml_backend_buffer_type_t buft) {
    if (orig->iface.get_name == ggml_backend_mpi_buffer_type_name) {
        ((ggml_backend_mpi_buffer_type_context*)(orig->context))->wrapped_buffer_type = buft;
    } else {
        GGML_ASSERT(!"Original buffer type must be an MPI buffer type.");
    }

    return orig;

}

ggml_backend_buffer_t ggml_backend_mpi_set_wrapped_buffer(ggml_backend_buffer_t orig, ggml_backend_buffer_t buf) {
    GGML_ASSERT(buf != nullptr);
    GGML_ASSERT(buf->iface.get_name != ggml_backend_mpi_buffer_name);
    if (orig->iface.get_name == ggml_backend_mpi_buffer_name) {
        ((ggml_backend_mpi_buffer_context*)(orig->context))->wrapped_buffer = buf;
        if (orig->buft != nullptr) {
            ggml_backend_mpi_buffer_type_set_wrapped_buffer_type(orig->buft, buf->buft);
        }
    } else {
        fprintf(stderr, "Original buffer name: %s\n", ggml_backend_buffer_name(orig));
        GGML_ASSERT(!"Original buffer must be an MPI buffer.");

    }
    return orig;
}

GGML_CALL static enum ggml_status ggml_backend_mpi_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {



    auto * ctx = (ggml_mpi_context *) backend->context;

    if (ctx->remote || ctx->size < 1 || ctx->rank < 0) {
        return GGML_STATUS_SUCCESS;
    }

    std::vector<std::pair<ggml_backend_buffer_t, std::vector<ggml_backend_buffer_t>>> old_buffs;
    old_buffs.resize(cgraph->n_nodes);
    std::vector<ggml_backend_buffer_t> old_view_buffs;
    old_view_buffs.resize(cgraph->n_nodes);


    for (int i = 0; i < cgraph->n_nodes; i++) {
        old_buffs[i].first = cgraph->nodes[i]->buffer;
        old_buffs[i].second = std::vector<ggml_backend_buffer_t>();
        old_buffs[i].second.resize(GGML_MAX_SRC);

        for (size_t j = 0; j < GGML_MAX_SRC; j++) {
            auto &src = cgraph->nodes[i]->src[j];
            if (src == nullptr || src->buffer == nullptr) {
                break;
            }
//            fprintf(stderr, "Previous source: %s\n", src->name);
            old_buffs[i].second[j] = src->buffer;

        }

        auto *src = cgraph->nodes[i]->view_src;
        if (src != nullptr) {
            if (src->buffer->buft != nullptr) {
                old_view_buffs[i] = src->buffer;

            }
        }
    }

    size_t n_srcs = 0;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (cgraph->nodes[i]->buffer->buft->iface.get_name == ggml_backend_mpi_buffer_type_name) {
            cgraph->nodes[i]->buffer = ggml_backend_mpi_buffer_unwrap(cgraph->nodes[i]->buffer);
        }

        for (auto &src: cgraph->nodes[i]->src) {
            if (src == nullptr) {
                break;
            }
            if (src->buffer->buft->iface.get_name == ggml_backend_mpi_buffer_type_name) {
                n_srcs++;
                src->buffer = ggml_backend_mpi_buffer_unwrap(src->buffer);
//                fprintf(stderr, "After unwrapping source: %s\n", src->name);

            }
        }

        auto *src = cgraph->nodes[i]->view_src;
        if (src != nullptr) {
            if (src->buffer->buft != nullptr) {

                if (src->buffer->buft->iface.get_name == ggml_backend_mpi_buffer_type_name) {
                    n_srcs++;
                    src->buffer = ggml_backend_mpi_buffer_unwrap(src->buffer);
                }
            }
        }
    }
    std::vector<ggml_backend_buffer_type_t> old_buffs_leaves;
    for (int i = 0; i < cgraph->n_leafs; i++) {
        old_buffs_leaves.push_back(cgraph->leafs[i]->buffer->buft);
        if (cgraph->leafs[i]->buffer->buft->iface.get_name == ggml_backend_mpi_buffer_type_name) {
            cgraph->leafs[i]->buffer = ggml_backend_mpi_buffer_unwrap(cgraph->leafs[i]->buffer);
        }
    }

    // TODO exploding memory usage cause we replace the buffer with the wrapped buffer,
    //  but don't free the contexts, and then create new ones when we re-wrap


    if (!ctx->remote && !ctx->backends.empty()) {
        ggml_backend_sched_t sched = ggml_backend_sched_new(ctx->backends.data(), nullptr,
                                                            (int) ctx->backends.size(), cgraph->n_nodes + cgraph->n_leafs + n_srcs, false);

//        ggml_backend_sched_reserve(sched, cgraph);
        ggml_backend_sched_graph_compute(sched, cgraph);
        ggml_backend_sched_free(sched);

    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (cgraph->nodes[i]->buffer->iface.get_name != ggml_backend_mpi_buffer_name) {
            cgraph->nodes[i]->buffer = ggml_backend_mpi_set_wrapped_buffer(old_buffs[i].first, cgraph->nodes[i]->buffer);
        }


        for (int iter = 0; iter < GGML_MAX_SRC; iter++) {
            auto* src_node = cgraph->nodes[i]->src[iter];
            if (src_node == nullptr || src_node->buffer == nullptr) {
                break;
            }

//            fprintf(stderr, "After compute src: %s\n", src_node->name);

            if (src_node->buffer->iface.get_name == ggml_backend_mpi_buffer_name) {
                continue;
            }

            src_node->buffer = ggml_backend_mpi_set_wrapped_buffer(old_buffs[i].second[iter], src_node->buffer);

//            fprintf(stderr, "After setting wrapped buffer src: %s\n", src_node->name);

        }
        if(cgraph->nodes[i]->view_src != nullptr && cgraph->nodes[i]->view_src->buffer->buft != nullptr) {

            if (old_view_buffs[i] != nullptr) {
                if (old_view_buffs[i]->iface.get_name == ggml_backend_mpi_buffer_name && cgraph->nodes[i]->view_src->buffer->iface.get_name != ggml_backend_mpi_buffer_name) {
                    cgraph->nodes[i]->view_src->buffer = ggml_backend_mpi_set_wrapped_buffer(old_view_buffs[i], cgraph->nodes[i]->view_src->buffer);
                }
            }
        }

    }


    // FIXME check if this is correct or not (it's probably not)
    for (int i = 0; i < cgraph->n_leafs; i++) {
        GGML_ASSERT(false);
//        cgraph->leafs[i]->buffer = ggml_backend_mpi_wrap_buffer(cgraph->leafs[i]->buffer);
//        ggml_backend_mpi_buffer_type_set_rank(cgraph->leafs[i]->buffer->buft, ctx->rank);
    }

    return GGML_STATUS_SUCCESS;
}




static void ggml_backend_mpi_free(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_mpi_context *>(backend->context);

    delete ctx;


    delete backend;
}

static ggml_backend_buffer_type_t ggml_backend_mpi_get_default_buffer_type(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_mpi_context *>(backend->context);


    auto * buft = ggml_backend_mpi_wrap_buffer_type(ctx->backends.front()->iface.get_default_buffer_type(ctx->backends.front()));
    ggml_backend_mpi_buffer_type_copy_ctx_from_backend(buft, backend);

    return buft;
}

GGML_CALL static bool ggml_backend_mpi_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    return ggml_backend_supports_op(((ggml_mpi_context *) backend->context)->backends.front(),op);
}




GGML_CALL bool ggml_backend_is_mpi(ggml_backend_t backend) {
    return backend && backend->iface.get_name == ggml_backend_mpi_name;
}

bool ggml_mpi_test_recv(ggml_backend_t backend) {
    if (!ggml_backend_is_mpi(backend)) {
        return false;
    }

    auto * ctx_mpi = (ggml_mpi_context*)backend->context;
    if (ctx_mpi->size < 1 || ctx_mpi->rank < 0) {
        return false;
    }
    if (ctx_mpi->requests.size() < 2) {
//        fprintf(stderr, "MPI REQUESTS EMPTY\n");
        return false;
    }

    int flag;
    MPI_Test(&(*(ctx_mpi->requests.begin()+1)).logits_receive_request, &flag, MPI_STATUS_IGNORE);
//    fprintf(stderr, "TESTING FOR RECEIVED: %d\n", flag);
    return flag;
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_mpi_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);
GGML_CALL static size_t ggml_backend_mpi_buffer_type_get_alignment(ggml_backend_buffer_type_t buft);
GGML_CALL static bool ggml_backend_mpi_buffer_type_supports_backend(ggml_backend_buffer_type_t buft, ggml_backend_t backend);
const static auto EMPTY_BUFFER_TYPE = new ggml_backend_buffer_type{
        /* .iface    = */ {
                                  /* .get_name         = */ ggml_backend_mpi_buffer_type_name,
                                  /* .alloc_buffer     = */ ggml_backend_mpi_buffer_type_alloc_buffer,
                                  /* .get_alignment    = */ ggml_backend_mpi_buffer_type_get_alignment,
                                  /* .get_max_size     = */ nullptr,
                                  /* .get_alloc_size   = */ nullptr,
                                  /* .supports_backend = */ ggml_backend_mpi_buffer_type_supports_backend,
                                  /* .is_host          = */ nullptr,
                          },
        /* .context  = */ new ggml_backend_mpi_buffer_type_context{
                /* .name                = */ "MPI",
                /* .wrapped_buffer_type = */ nullptr,
                /* .ctx_mpi             = */ ggml_mpi_init(),
                /* .empty               = */ true
        }
};



GGML_CALL static ggml_backend_buffer_t ggml_backend_mpi_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
//    fprintf(stderr, "ALLOCATING BUFFER WITH COMM SIZE %zu AND RANK %d\n", ggml_backend_mpi_buffer_type_ctx(buft)->size, ggml_backend_mpi_buffer_type_ctx(buft)->rank);
    GGML_ASSERT(ggml_backend_mpi_buffer_type_ctx(buft)->size < 3 && ggml_backend_mpi_buffer_type_ctx(buft)->rank < 2);
    auto* buffer = ggml_backend_mpi_wrap_buffer(
            ggml_backend_buft_alloc_buffer(ggml_backend_mpi_buffer_type_unwrap(buft), size)
            );

    ggml_backend_mpi_buffer_copy_ctx_from_type(buft, buffer);

    return buffer;
}

GGML_CALL static size_t ggml_backend_mpi_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_get_alignment(ggml_backend_mpi_buffer_type_unwrap(buft));
}

GGML_CALL static size_t ggml_backend_mpi_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_get_max_size(ggml_backend_mpi_buffer_type_unwrap(buft));
}

GGML_CALL static size_t ggml_backend_mpi_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    // Have to do this instead of calling ggml_backend_type_get_alloc_size because that signature doesn't have const on tensor
    size_t ret = ggml_backend_mpi_buffer_type_unwrap(buft)->iface.get_alloc_size(ggml_backend_mpi_buffer_type_unwrap(buft), tensor);
    return ret;
}

GGML_CALL static bool ggml_backend_mpi_buffer_type_supports_backend(ggml_backend_buffer_type_t buft, ggml_backend_t backend) {
    return backend != nullptr && ggml_backend_is_mpi(backend) && ggml_backend_mpi_buffer_type_rank(buft) == ggml_backend_mpi_rank(backend)
        && ggml_backend_mpi_buffer_type_get_comm(buft) == ggml_backend_mpi_get_comm(backend)
        && ggml_backend_buft_supports_backend(ggml_backend_mpi_buffer_type_unwrap(buft), ((ggml_mpi_context*)backend->context)->backends.front());
}

GGML_CALL static bool ggml_backend_mpi_buffer_type_is_host(ggml_backend_buffer_type_t buft) {

    return ggml_backend_mpi_buffer_type_rank(buft) == ggml_backend_mpi_buffer_type_local_rank(buft) && ggml_backend_buft_is_host(ggml_backend_mpi_buffer_type_unwrap(buft));
}



GGML_CALL ggml_backend_buffer_type_t ggml_backend_mpi_wrap_buffer_type(ggml_backend_buffer_type_t buft) {

    GGML_ASSERT(buft->iface.get_name != ggml_backend_mpi_buffer_type_name);


    ggml_backend_buffer_type_i ggml_backend_mpi_buffer_type_interface = {
            /* .get_name         = */ ggml_backend_mpi_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_mpi_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_mpi_buffer_type_get_alignment,
            /* .get_max_size     = */ (buft->iface.get_max_size != nullptr ) ? ggml_backend_mpi_buffer_type_get_max_size : nullptr,
            /* .get_alloc_size   = */ (buft->iface.get_alloc_size != nullptr ) ? ggml_backend_mpi_buffer_type_get_alloc_size : nullptr,
            /* .supports_backend = */ ggml_backend_mpi_buffer_type_supports_backend,
            /* .is_host          = */ (buft->iface.is_host != nullptr ) ? ggml_backend_mpi_buffer_type_is_host : nullptr,
    };



    auto* ggml_backend_wrapped_buffer_type = new ggml_backend_buffer_type {
            /* .iface    = */ ggml_backend_mpi_buffer_type_interface,
            /* .context  = */ new ggml_backend_mpi_buffer_type_context{
                                /* .name                = */ "MPI",
                                /* .wrapped_buffer_type = */ buft,
                                /* .ctx_mpi             = */ ggml_mpi_init(),
                                /* .empty               = */ false
                            }
    };

    // Set rank to 0 as default
    ggml_backend_mpi_buffer_type_set_rank(ggml_backend_wrapped_buffer_type, 0);

    return ggml_backend_wrapped_buffer_type;
}

GGML_CALL ggml_backend_buffer_type_t ggml_backend_mpi_wrap_buffer_type_with_backend(ggml_backend_buffer_type_t buft, ggml_backend_t backend) {
    auto * wrapped = ggml_backend_mpi_wrap_buffer_type(buft);
    ggml_backend_mpi_buffer_type_copy_ctx_from_backend(wrapped, backend);
    return wrapped;
}



GGML_CALL static void * ggml_backend_mpi_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * ctx = (ggml_backend_mpi_buffer_context *) buffer->context;
    return ctx->wrapped_buffer->iface.get_base(ctx->wrapped_buffer);
}

GGML_CALL static void ggml_backend_mpi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    if (!((ggml_backend_mpi_buffer_context*)buffer->context)->empty) {
        auto *ctx = (ggml_backend_mpi_buffer_context *) buffer->context;
        ctx->wrapped_buffer->iface.free_buffer(ctx->wrapped_buffer);
    }
}

GGML_CALL static void ggml_backend_mpi_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * ctx = (ggml_backend_mpi_buffer_context *) buffer->context;

    if (ggml_backend_mpi_buffer_rank(buffer) != ggml_backend_mpi_buffer_local_rank(buffer)) {
//        fprintf(stderr, "IGNORING SET_TENSOR FOR TENSOR %s, BUFFER %s\n", tensor->name, ggml_backend_buffer_name(buffer));
        return;
    }

//    fprintf(stderr, "SETTING TENSOR WITHOUT MPI CALLS FOR %s (%s) AND TGT BUFFER %s\n", tensor->name, ggml_backend_buffer_name(tensor->buffer), ggml_backend_buffer_name(buffer));
    ctx->wrapped_buffer->iface.set_tensor(ctx->wrapped_buffer, tensor, data, offset, size);
}

GGML_CALL static void ggml_backend_mpi_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    int rank = ggml_backend_mpi_buffer_local_rank(tensor->buffer);

    int src_rank = ggml_backend_mpi_buffer_rank(tensor->buffer);

//    if (ggml_backend_mpi_buffer_rank(buffer) != ggml_backend_mpi_buffer_local_rank(buffer)) {
//        return;
//    }

    if (rank != src_rank) {
//        fprintf(stderr, "Getting tensor synchronous: %s, buffer %s\n", tensor->name, ggml_backend_buffer_name(buffer));
        ggml_mpi_tensor_recv(tensor, data, ggml_backend_mpi_buffer_rank(tensor->buffer), ggml_backend_mpi_buffer_get_comm(tensor->buffer));
        return;
    }

    ggml_backend_mpi_buffer_unwrap(buffer)->iface.get_tensor(ggml_backend_mpi_buffer_unwrap(buffer), tensor, data, offset, size);
}

GGML_CALL static bool ggml_backend_mpi_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (ggml_backend_mpi_buffer_rank(src->buffer) == ggml_backend_mpi_buffer_rank(dst->buffer) && ggml_backend_mpi_buffer_local_rank(buffer) == ggml_backend_mpi_buffer_rank(src->buffer)) {
        return ggml_backend_mpi_buffer_unwrap(buffer)->iface.cpy_tensor(ggml_backend_mpi_buffer_unwrap(buffer), src,
                                                                        dst);
    }

//    fprintf(stderr, "IGNORING SYNC COPY TENSOR FROM %s TO %s WITH BUFFER %s\n", src->name, dst->name, ggml_backend_buffer_name(buffer));

    return true;
}

GGML_CALL static void ggml_backend_mpi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    if (!ggml_backend_mpi_buffer_ctx(buffer)->remote) {
        ggml_backend_mpi_buffer_unwrap(buffer)->iface.clear(ggml_backend_mpi_buffer_unwrap(buffer), value);
    }
}

GGML_CALL static void ggml_backend_mpi_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
//    fprintf(stderr, "Init tensor with buffer %s, tensor %s, tensor buffer %s, tensor view src %s, tensor vs buff %s\n",
//            ggml_backend_buffer_name(buffer), tensor->name, ggml_backend_buffer_name(tensor->buffer), tensor->view_src !=
//                    nullptr ? tensor->view_src->name : "", tensor->view_src != nullptr ? ggml_backend_buffer_name(tensor->view_src->buffer) : "");

    auto *orig_buffer = tensor->buffer;
    if (ggml_backend_mpi_buffer_ctx(buffer)->remote) {
        return;
    }
    tensor->buffer = ggml_backend_mpi_buffer_unwrap(tensor->buffer);

    bool view_src_null = tensor->view_src == nullptr;
    ggml_backend_buffer_t orig_view_src_buffer = nullptr;
    if (!view_src_null) {
         orig_view_src_buffer = tensor->view_src->buffer;
        tensor->view_src->buffer = ggml_backend_mpi_buffer_unwrap(tensor->view_src->buffer);
    }

    std::vector<ggml_backend_buffer_t> orig_src_buffers(0);
    for (auto & src : tensor->src) {
        if (src == nullptr) {
            break;
        }


        orig_src_buffers.push_back(src->buffer);

        if (src->buffer != nullptr && src->buffer->iface.get_name == ggml_backend_mpi_buffer_name) {
            src->buffer = ggml_backend_mpi_buffer_unwrap(src->buffer);
        }
    }


    ggml_backend_buffer_init_tensor(ggml_backend_mpi_buffer_unwrap(buffer), tensor);
    tensor->buffer = ggml_backend_mpi_set_wrapped_buffer(orig_buffer, tensor->buffer);
    if (!view_src_null) {
        tensor->view_src->buffer = ggml_backend_mpi_set_wrapped_buffer(orig_view_src_buffer, tensor->view_src->buffer);
    }

    for (size_t i = 0; i < orig_src_buffers.size(); i++) {
        if (orig_src_buffers[i]->iface.get_name == ggml_backend_mpi_buffer_name) {
            tensor->src[i]->buffer = ggml_backend_mpi_set_wrapped_buffer(orig_src_buffers[i], tensor->src[i]->buffer);
        }
    }
}



GGML_CALL ggml_backend_buffer_t ggml_backend_mpi_wrap_buffer_with_backend(ggml_backend_buffer_t buf, ggml_backend_t backend) {
    auto * wrapped = ggml_backend_mpi_wrap_buffer(buf);
    ggml_backend_mpi_buffer_copy_ctx_from_backend(wrapped, backend);
    return wrapped;

}

GGML_CALL ggml_backend_buffer_t ggml_backend_mpi_wrap_buffer(ggml_backend_buffer_t buf) {

    struct ggml_backend_buffer_i mpi_backend_buffer_i = {
            /* .get_name        = */ ggml_backend_mpi_buffer_name,
            /* .free_buffer     = */ ggml_backend_mpi_buffer_free_buffer,
            /* .get_base        = */ ggml_backend_mpi_buffer_get_base,
            /* .init_tensor     = */ (buf != nullptr && buf->iface.init_tensor != nullptr) ? ggml_backend_mpi_buffer_init_tensor : nullptr,
            /* .set_tensor      = */ ggml_backend_mpi_buffer_set_tensor,
            /* .get_tensor      = */ ggml_backend_mpi_buffer_get_tensor,
            /* .cpy_tensor      = */ ggml_backend_mpi_buffer_cpy_tensor,
            /* .clear           = */ ggml_backend_mpi_buffer_clear,
            /* .reset           = */ nullptr,
    };

//    if (cached_buffer_wrappers.find(buf) != cached_buffer_wrappers.end()) {
//        fprintf(stderr, "Returning cached buffer with name %s\n", cached_buffer_wrappers[buf]->iface.get_name(cached_buffer_wrappers[buf]));
//        auto * ret = new ggml_backend_buffer;
//        *ret = *cached_buffer_wrappers[buf];
//        auto * ret_type = new ggml_backend_buffer_type;
//        *ret_type = *ret->buft;
//        ret->buft = ret_type;
//        return ret;
//    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//    fprintf(stderr, "Wrapping buffer %s at rank %d\n", ggml_backend_buffer_name(buf), rank);

    if (buf != nullptr && buf->iface.get_name == ggml_backend_mpi_buffer_name) {
        fprintf(stderr, "WRAPPING AN ALREADY WRAPPED BUFFER: %s\n", ggml_backend_buffer_name(buf));
        GGML_ASSERT(false);
    }

    ggml_backend_buffer_type_t t = (buf != nullptr) ? ggml_backend_mpi_wrap_buffer_type(buf->buft) : EMPTY_BUFFER_TYPE;

    auto *buffer = new ggml_backend_buffer {
            /* .interface = */ mpi_backend_buffer_i,
            /* .buft      = */ t,
            /* .context   = */ new ggml_backend_mpi_buffer_context{
                                buf, ggml_mpi_init(), buf == nullptr},
            /* .size      = */ (buf != nullptr) ? buf->size : 0,
            /* .usage     = */ (buf != nullptr) ? buf->usage : GGML_BACKEND_BUFFER_USAGE_ANY
    };

    // Default to node 0 when wrapping buffers
    if (buf != nullptr) {
        ggml_backend_mpi_buffer_set_rank(buffer, 0);
    } else {
        ggml_backend_mpi_buffer_set_rank(buffer, -1);
        ggml_backend_mpi_buffer_ctx(buffer)->comm = MPI_COMM_NULL;
        ggml_backend_mpi_buffer_ctx(buffer)->size = 0;

    }


    return buffer;
}

bool ggml_backend_mpi_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, struct ggml_tensor * src, struct ggml_tensor * dst) {
    int src_rank = ggml_backend_mpi_buffer_rank(src->buffer);
    int dst_rank = ggml_backend_mpi_buffer_rank(dst->buffer);

    auto * src_ctx = static_cast<ggml_mpi_context *>(backend_src->context);
    auto * dst_ctx = static_cast<ggml_mpi_context *>(backend_dst->context);


    if (src_ctx->remote && dst_ctx->remote) {
        return true;
    }

    if (src_rank == dst_rank) {
        src->buffer = ggml_backend_mpi_buffer_unwrap(src->buffer);
        if (src->view_src) {
            src->view_src->buffer = ggml_backend_mpi_buffer_unwrap(src->view_src->buffer);
        }
        dst->buffer = ggml_backend_mpi_buffer_unwrap(dst->buffer);
        if (dst->view_src) {
            dst->view_src->buffer = ggml_backend_mpi_buffer_unwrap(dst->view_src->buffer);
        }
        ggml_backend_tensor_copy_async(((ggml_mpi_context *) backend_src->context)->backends.front(),((ggml_mpi_context *) backend_dst->context)->backends.front(), src, dst);

        src->buffer = ggml_backend_mpi_wrap_buffer_with_backend(src->buffer, backend_src);
        if (src->view_src) {
            src->view_src->buffer = ggml_backend_mpi_wrap_buffer_with_backend(src->view_src->buffer, backend_src);
        }
        dst->buffer = ggml_backend_mpi_wrap_buffer_with_backend(dst->buffer, backend_dst);
        if (dst->view_src) {
            dst->view_src->buffer = ggml_backend_mpi_wrap_buffer_with_backend(dst->view_src->buffer, backend_dst);
        }
//        src->buffer->iface.cpy_tensor(src->buffer, src, dst);
        return true;
    }

    if (src_rank == ggml_backend_mpi_local_rank(backend_src)) {
        ggml_mpi_tensor_send(src, dst_rank, dst_ctx->comm);
    } else if (dst_rank == ggml_backend_mpi_local_rank(backend_dst)){
//        fprintf(stderr, "DOING SYNCHRONOUS RECEIVE FOR ASYNC COPY\n");
        ggml_mpi_tensor_recv(dst, src_rank, src_ctx->comm);
    }
//    fprintf(stderr, "ATTEMPTING ASYNC COPY FOR SRC TENSOR %s TO DST TENSOR %s WITH SRC BACKEND %s AND DST BACKEND %s\n", src->name, dst->name, ggml_backend_name(backend_src), ggml_backend_name(backend_dst));
    return true;

}

void ggml_backend_mpi_set_tensor_async(ggml_backend_t backend, struct ggml_tensor * dst, const void* data, size_t offset, size_t size) {
    int dst_rank = ggml_backend_mpi_buffer_rank(dst->buffer);


    auto * ctx = static_cast<ggml_mpi_context *>(backend->context);

    GGML_ASSERT(ctx->rank == dst_rank);

    if (dst_rank == ggml_backend_mpi_buffer_local_rank(dst->buffer)) {
        auto * old_buffer = dst->buffer;
        dst->buffer = ggml_backend_mpi_buffer_unwrap(dst->buffer);
        if (dst->view_src) {
            dst->view_src->buffer = ggml_backend_mpi_buffer_unwrap(dst->view_src->buffer);
        }
        ggml_backend_tensor_set_async(((ggml_mpi_context *) backend->context)->backends.front(), dst, data, offset, size);
        dst->buffer = ggml_backend_mpi_wrap_buffer_with_backend(dst->buffer, backend);
        if (dst->view_src) {
            dst->view_src->buffer = ggml_backend_mpi_wrap_buffer_with_backend(dst->view_src->buffer, backend);
        }
//        dst->buffer = old_buffer;
    } else {

        ggml_mpi_tensor_send(dst, data, ctx->rank, ctx->comm);
    }


}

GGML_CALL static void ggml_backend_mpi_get_tensor_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    // We can assume that the tensor will be on CPU
    int rank = ggml_backend_mpi_buffer_local_rank(tensor->buffer);

    int src_rank = ggml_backend_mpi_buffer_rank(tensor->buffer);

//    if (ggml_backend_mpi_buffer_rank(buffer) != ggml_backend_mpi_buffer_local_rank(buffer)) {
//        return;
//    }

    if (rank != src_rank) {
//        fprintf(stderr, "Receiving tensor async: %s, buffer %s, backend %s\n", tensor->name, ggml_backend_buffer_name(tensor->buffer), ggml_backend_name(backend));
        ggml_mpi_tensor_recv_async((ggml_mpi_context*)backend->context, tensor, data, ggml_backend_mpi_buffer_rank(tensor->buffer), ggml_backend_mpi_buffer_get_comm(tensor->buffer));
        return;
    }

//    fprintf(stderr, "Getting tensor async: %s, buffer %s\n", tensor->name, ggml_backend_buffer_name(tensor->buffer));

    ggml_backend_tensor_get(tensor, data, offset, size);

    if (rank == (int)((ggml_mpi_context*)backend->context)->size - 1 && rank != 0) {
        ggml_mpi_tensor_send(tensor, 0, ggml_backend_mpi_get_comm(backend));

    }

}

GGML_CALL void ggml_backend_mpi_pop_request(ggml_backend_t backend) {
    if (!ggml_backend_is_mpi(backend)) {
        return;
    }

    auto * ctx_mpi = (ggml_mpi_context*)backend->context;

    if (!ctx_mpi->requests.empty() && ctx_mpi->requests.front().received) {
//        fprintf(stderr, "POPPING REQUEST FOR BACKEND %s\n", ggml_backend_name(backend));
        ctx_mpi->requests.pop_front();
    }
}

GGML_CALL static void ggml_backend_mpi_synchronize(ggml_backend_t backend) {
    if (!ggml_backend_is_mpi(backend)) {
        return;
    }

    auto * ctx_mpi = (ggml_mpi_context*)backend->context;

    if (!ctx_mpi->requests.empty() && !ctx_mpi->requests.front().received) {
//        fprintf(stderr, "DOING MPI WAIT ON BACKEND %s\n", ggml_backend_name(backend));
        MPI_Wait(&ctx_mpi->requests.front().logits_receive_request, MPI_STATUS_IGNORE);
        ctx_mpi->requests.front().received = true;
//        fprintf(stderr, "FINISHED MPI WAIT ON BACKEND %s\n", ggml_backend_name(backend));

    }

    if (!ctx_mpi->remote) {
//        fprintf(stderr, "DOING NON-MPI BACKEND SYNC: %s\n", ggml_backend_name(backend));
        ggml_backend_synchronize(((ggml_mpi_context*)backend->context)->backends.front());
    }
}

ggml_backend_t ggml_backend_mpi_init(ggml_mpi_context * ctx_mpi, ggml_backend_t * wrapped_backends, size_t num_backends, int rank) {

    static ggml_guid backend_mpi_guid = {0xec, 0x39, 0xce, 0x40, 0xc3, 0x43, 0x49, 0x36, 0x96, 0x03, 0x55, 0x77, 0x5c, 0x1f, 0x44, 0xd3};


    ggml_mpi_context * ctx = ggml_mpi_init();
    ctx->comm = ctx_mpi->comm;
    if (ctx->comm != MPI_COMM_NULL) {
        int size = 0;
        MPI_Comm_rank(ctx->comm, &ctx->rank);
        MPI_Comm_size(ctx->comm, &size);
        ctx->size = size;
    } else {
//        fprintf(stderr, "CREATING NEW BACKEND WITH NULL COMM ON RANK %d\n", ctx_mpi->rank);
        ctx->rank = -1;
        ctx->size = 0;
    }
    std::vector<ggml_backend_t> wrapped_backends_v;
    for (size_t i = 0; i < num_backends; i++) {
        wrapped_backends_v.push_back(wrapped_backends[i]);
    }
    if (ctx->rank == rank && ctx->comm != MPI_COMM_NULL) {

    } else {
        ctx->remote = true;
    }
    ctx->backends = wrapped_backends_v;
    ctx->rank = rank;
    struct ggml_backend_i mpi_backend_i = {
            /* .get_name                = */ ggml_backend_mpi_name,
            /* .free                    = */ ggml_backend_mpi_free,
            /* .get_default_buffer_type = */ ggml_backend_mpi_get_default_buffer_type,
            /* .set_tensor_async        = */ ggml_backend_mpi_set_tensor_async,
            /* .get_tensor_async        = */ ggml_backend_mpi_get_tensor_async,
            /* .cpy_tensor_async        = */ ggml_backend_mpi_cpy_tensor_async,
            /* .synchronize             = */ ggml_backend_mpi_synchronize,
            /* .graph_plan_create       = */ nullptr,
            /* .graph_plan_free         = */ nullptr,
            /* .graph_plan_compute      = */ nullptr,
            /* .graph_compute           = */ ggml_backend_mpi_graph_compute,
            /* .supports_op             = */ ggml_backend_mpi_supports_op,
            /* .offload_op              = */ nullptr,
            /* .event_new               = */ nullptr,
            /* .event_free              = */ nullptr,
            /* .event_record            = */ nullptr,
            /* .event_wait              = */ nullptr,
            /* .event_synchronize       = */ nullptr,
    };

    auto *mpi_backend = new ggml_backend {
            /* .guid      = */ &backend_mpi_guid,
            /* .interface = */ mpi_backend_i,
            /* .context   = */ ctx,
    };

    return mpi_backend;
}



GGML_CALL void ggml_backend_mpi_buffer_type_set_rank(ggml_backend_buffer_type_t buft, int rank) {
    if (buft->iface.get_name == ggml_backend_mpi_buffer_type_name) {
        if (((ggml_backend_mpi_buffer_type_context *) buft->context)->ctx_mpi->rank >= 0) {
            ((ggml_backend_mpi_buffer_type_context *) buft->context)->ctx_mpi->rank = rank;
        }
    } else {
        GGML_ASSERT(!"Buffer type must be wrapped in ggml_backend_mpi_buffer_type");
    }
}

GGML_CALL void ggml_backend_mpi_buffer_set_rank(ggml_backend_buffer_t buf, int rank) {
    if (buf->iface.get_name == ggml_backend_mpi_buffer_name) {
        if (((ggml_backend_mpi_buffer_context *) buf->context)->ctx_mpi->rank >= 0) {
            ((ggml_backend_mpi_buffer_context *) buf->context)->ctx_mpi->rank = rank;
            ggml_backend_mpi_buffer_type_set_rank(buf->buft, rank);
        }
    } else {
        GGML_ASSERT(!"Buffer type must be wrapped in ggml_backend_mpi_buffer_type");
    }
}

GGML_CALL void ggml_backend_mpi_buffer_type_init_with_context(ggml_backend_buffer_type_t buft, ggml_mpi_context * context) {
    auto * src_ctx = ((ggml_backend_mpi_buffer_type_context*)buft->context)->ctx_mpi;
    src_ctx->comm = context->comm;
    src_ctx->size = context->size;
    src_ctx->remote = context->remote;

}

GGML_CALL void ggml_backend_mpi_buffer_init_with_context(ggml_backend_buffer_t buff, ggml_mpi_context * context) {
    auto * src_ctx = ((ggml_backend_mpi_buffer_context*)buff->context)->ctx_mpi;
    src_ctx->comm = context->comm;
    src_ctx->size = context->size;
    src_ctx->remote = context->remote;
    ggml_backend_mpi_buffer_type_init_with_context(buff->buft, context);

}




